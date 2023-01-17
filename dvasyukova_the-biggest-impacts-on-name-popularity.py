%matplotlib inline

import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import cm

from bokeh.plotting import figure, output_notebook, show, ColumnDataSource

from bokeh.models import HoverTool, LabelSet, FixedTicker

from bokeh.charts import Bar

from IPython.display import HTML, display



output_notebook()
df = pd.read_csv('../input/NationalNames.csv', index_col='Id')
df['Popularity'] = df.Count.values*1000000/df.groupby(['Year','Gender'])['Count'].transform(sum).values
def popularity_diff(group):

    yd = group.Year.diff().fillna(0)

    popd = group.Popularity.diff()

    popd.loc[yd>1] = group.Popularity.loc[yd>1]

    popd.iloc[0] = group.Popularity.iloc[0] if group.Year.iloc[0] > 1880 else 0

    return popd



df['PopDiff'] = df.groupby(['Name','Gender']).apply(popularity_diff).reset_index().set_index('Id')['Popularity']
df['TrendingRank'] = df.groupby(['Year','Gender'])['PopDiff'].rank(ascending=False)

df['PopularityRank'] = df.groupby(['Year','Gender'])['Popularity'].rank(ascending=False)
gr = df.groupby(['Name','Gender'])

names = gr.agg({'PopDiff':{'MaxPopDiff':'max',

                           'MinPopDiff':'min'},

                'Count':{'TotalCount':'sum'},

                'Year':{'FirstYear':'min',

                        'LastYear':'max',

                        'NYears':'count'},

                'Popularity':{'MaxPopularity':'max'}})

names.columns = names.columns.droplevel(0)

def bestyear(group, col):

    years, counts = group['Year'].values, group[col].values

    ind = np.argmax(counts)

    return years[ind]

names['BestYearPop'] = gr.apply(bestyear,'Popularity')

names['BestYearPopDiff'] = gr.apply(bestyear,'PopDiff')

names = names.sort_values(by='MaxPopDiff',ascending=False)
data = names.head(300).reset_index()

data['Rank'] = data.groupby('BestYearPopDiff')['MaxPopDiff'].rank(ascending=False)

data['size'] = 6 + np.log(data.MaxPopDiff)

data['alpha'] = np.clip(0.1+data.MaxPopDiff/data.MaxPopDiff.max(),0,1)

data['color'] = 'blue'

data.loc[data.Gender=='F','color']='red'

data['text_color']='#555555'

data['cause'] = ''

data.loc[data.Name=='Linda','cause'] = '"Linda" is a popular song written about then one year old future star Linda McCartney. It was written by Jack Lawrence, and published in 1946.'

data.loc[data.Name=='Shirley','cause'] = 'Shirley Temple was a child actress wildly popular since 1935 for films Bright Eyes, Curly Top and Heidi'

data.loc[data.Name.isin(['Michelle','Michele']),'cause'] = '"Michelle" is a love ballad by the Beatles. It is featured on their Rubber Soul album, released in December 1965. "Michelle" won the Grammy Award for Song of the Year in 1967 and has since become one of the best known and often recorded of all Beatles songs.'

data.loc[data.Name=='Amanda','cause']='"Amanda" is a 1973 song written by Bob McDill and recorded by both Don Williams (1973) and Waylon Jennings (1974). In April 1979 the song was issued as a single, and it soon became one of the biggest country hits of 1979.'

data.loc[data.Name.isin(['Jaime','Jamie']),'cause']='Jaime Sommers is an 1970s iconic television leading female science fiction action character who takes on special high-risk government missions using her superhuman bionic powers in the American television series The Bionic Woman (1976–1978).'

data.loc[data.Name.isin(['Katina','Catina']),'cause']='In 1972 the name Katina was used for a newborn baby on the soap opera "Where the Heart Is"'

data.loc[data.Name.isin(['Judy','Judith']),'cause']='Judy Garland stars as Dorothy in the Wizard of Oz movie (1939)'

data.loc[data.Name=='Whitney','cause']='The singer Whitney Houston was No. 1 artist of the year and her album was the No. 1 album of the year on the 1986 Billboard year-end charts.'

data.loc[data.Name=='Ashanti','cause']='In April 2002 the singer Ashanti released her eponymous debut album, which featured the hit song "Foolish", and sold over 503,000 copies in its first week of release throughout the U.S.'

data.loc[data.Name=='Woodrow','cause']='Woodrow Wilson ran for president of the USA in 1912.'

data.loc[data.Name=='Jacqueline','cause']='Jacqueline Kennedy becomes First Lady'

data.loc[data.cause!='','text_color'] = data['color']
source_noexpl = ColumnDataSource(data=data.loc[data.cause==''])

source_expl = ColumnDataSource(data=data.loc[data.cause!=''])



hover = HoverTool(

        tooltips="""

        <div>

            <div>

                <span style="font-size: 17px; font-weight: bold;">@Name</span>

                <span style="font-size: 15px; color: #966;">@BestYearPopDiff</span>

            </div>

            <div style='max-width: 300px'>

                <span style="font-size: 15px;">@cause</span>

            </div>

        </div>

        """

    )



p = figure(plot_width=800, plot_height=2000, tools=[hover,'pan'],

           title="Top {} trending names from {} to {}".format(data.shape[0],df.Year.min()+1, df.Year.max()))



p.circle('Rank', 'BestYearPopDiff', size='size', color='color',source=source_noexpl, alpha='alpha')

p.circle('Rank', 'BestYearPopDiff', size='size', color='color',source=source_expl, alpha='alpha')



labels_noexpl = LabelSet(x="Rank", y="BestYearPopDiff", text="Name", x_offset=8., y_offset=-7.,

                  text_font_size="10pt", text_color="text_color", text_font_style='normal',

                  source=source_noexpl, text_align='left')

labels_expl = LabelSet(x="Rank", y="BestYearPopDiff", text="Name", x_offset=8., y_offset=-7.,

                  text_font_size="10pt", text_color="text_color", text_font_style='bold',

                  source=source_expl, text_align='left')

p.add_layout(labels_noexpl)

p.add_layout(labels_expl)

p.yaxis[0].ticker=FixedTicker(ticks=np.arange(1880,2015,5))

show(p)
def report_name(name, gender):

    stats = names.loc[(name,gender)]

    html = """

    <p> {boygirl} name <strong>{name}</strong> has been in use for 

    {NYears:.0f} years from {FirstYear:.0f} to {LastYear:.0f}.</p>

    <p> It was most popular in {BestYearPop:.0f} when {MaxPopularity:.0f} babies in a million were named {name}.</p>

    <p> Its largest popularity raise was in <strong>{BestYearPopDiff:.0f}</strong> 

    with {MaxPopDiff:.0f} babies per million more named {name} than in previous year.</p>

    """.format(**{'boygirl':'Boy' if gender=='M' else 'Girl', 

                'name':name},

              **stats)

    display(HTML(html))

    data = df.loc[(df.Name==name)&(df.Gender==gender)]

    fig, ax = plt.subplots(2,1,figsize=(12,6),sharex=True,

                           gridspec_kw = {'height_ratios':[3, 1]})

    ax[0].bar(data.Year, data['Popularity'], width = 1, alpha=0.6,

           color = 'r' if gender=='F' else 'b')

    ax[0].set_ylabel('Babies per million')

    ax[1].semilogy(data.Year+0.5, data.TrendingRank,label='Trending rank')

    ax[1].semilogy(data.Year+0.5, data.PopularityRank,label='Popularity rank')

    ax[1].set_ylim(0.5,1100)

    ax[1].invert_yaxis()

    ax[1].set_yticklabels([str(int(x)) for x in ax[1].get_yticks()]);

    ax[1].legend()

    fig.suptitle(name,fontsize='large')

    return ax
impacts = []
ax = report_name('Linda','F')

ax[0].set_xlim(1930,1980);

ax[0].axvspan(1947+3/12,1947+3/12+13/52, alpha = 0.5, label='Linda song on the charts')

ax[0].legend(fontsize='large');
impacts.append({'Cause':'"Linda" song',

                'Year':1947,

                'Names':'Linda, Lynda',

                'PopularityGain':names.loc[[('Linda','F'),

                                            ('Lynda','F'),

                                            ('Linda','M')],'MaxPopDiff'].sum()})
ax = report_name('Shirley','F')

ax[0].set_xlim(1910,1970)
impacts.append({'Cause':'Shirley Temple, child actress',

                'Year':1935,

                'Names':'Shirley, Shirlee, Shrilie',

                'PopularityGain':names.loc[[('Shirley','F'),

                                            ('Shirlee','F'),

                                            ('Shirlie','F'),

                                            ('Shirley','M')],'MaxPopDiff'].sum()})
ax = report_name('Michelle','F')

ax[0].set_xlim(1940,2015)
impacts.append({'Cause':'"Michelle", Beatles song',

                'Year':1966,

                'Names':'Michelle, Michele',

                'PopularityGain':names.loc[[('Michelle','F'),

                                            ('Michele','F')],'MaxPopDiff'].sum()})
ax = report_name('Amanda','F')

ax[0].set_xlim(1960,2015);
impacts.append({'Cause':'"Amanda" song',

                'Year':1979,

                'Names':'Amanda',

                'PopularityGain':names.loc[[('Amanda','F')],'MaxPopDiff'].sum()})
ax=report_name('Jaime','F')

ax[0].set_xlim(1940,2015);
ax=report_name('Jamie','F')

ax[0].set_xlim(1940,2015);
impacts.append({'Cause':'"Bionic Woman" TV series',

                'Year':1976,

                'Names':'Jaime, Jamie',

                'PopularityGain':names.loc[[('Jaime','F'),

                                            ('Jamie','F'),

                                            ('Jami','F'),('Jaimie','F'),('Jayme','F'),('Jaimee','F'),

                                            ('Jamey','F'),('Jaymie','F'),('Jaimi','F'),('Jamy','F'),

                                            ('Jamye','F'),('Jaimy','F'),],'MaxPopDiff'].sum()})
ax=report_name('Judith','F')

ax[0].set_xlim(1920,1980);
ax=report_name('Judy','F')

ax[0].set_xlim(1920,1980);
impacts.append({'Cause':'Judy Garland in "Wizard of Oz"',

                'Year':1939,

                'Names':'Judith, Judy',

                'PopularityGain':names.loc[[('Judith','F'),

                                            ('Judy','F'),

                                            ('Judie','F')],'MaxPopDiff'].sum()})
ax=report_name('Katina','F')

ax[0].set_xlim(1960,2015);
impacts.append({'Cause':'Baby on "Where the Heart Is" soap opera',

                'Year':1972,

                'Names':'Katina, Catina',

                'PopularityGain':names.loc[[('Katina','F'),

                                            ('Catina','F')],'MaxPopDiff'].sum()})
ax=report_name('Whitney','F')

ax[0].set_xlim(1960,2015);
impacts.append({'Cause':'Whitney Houston, singer',

                'Year':1986,

                'Names':'Whitney',

                'PopularityGain':names.loc[[('Whitney','F')],'MaxPopDiff'].sum()})
ax=report_name('Ashanti','F')

ax[0].set_xlim(1980,2015);
impacts.append({'Cause':'Ashanti, singer',

                'Year':1986,

                'Names':'Ashanti',

                'PopularityGain':names.loc[[('Ashanti','F'),('Ashanty','F')],'MaxPopDiff'].sum()})
ax=report_name('Jacqueline','F')

ax[0].set_xlim(1920,2015);
ax=report_name('Jackie','F')

ax[0].set_xlim(1920,2015);
impacts.append({'Cause':'Jacqueline Kennedy, First Lady',

                'Year':1961,

                'Names':'Jacqueline, Jackie',

                'PopularityGain':names.loc[[('Jacqueline','F'),('Jackie','F'),('Jacquelyn','F'),

                                            ('Jacquline','F'),('Jacquelin','F'),('Jackqueline','F')],'MaxPopDiff'].sum()})
ax=report_name('Woodrow','M')

ax[0].set_xlim(1900,1960);
ax=report_name('Wilson','M')

ax[0].set_xlim(1900,1960);
impacts.append({'Cause':'Woodrow Wilson running for president',

                'Year':1912,

                'Names':'Woodrow, Wilson',

                'PopularityGain':names.loc[[('Woodrow','M'),('Wilson','M'),('Woodroe','M'),

                                            ('Woodrow','F')],'MaxPopDiff'].sum()})
pd.DataFrame(impacts).sort_values(by='PopularityGain',ascending=False).head(3)
res = pd.DataFrame(impacts).sort_values(by='PopularityGain')

fig, ax = plt.subplots(figsize=(10,6))

h = np.arange(res.shape[0])

ax.barh(h,res.PopularityGain)

ax.set_yticks(h+0.4)

ax.set_yticklabels(res.Cause.str.cat(res.Year.astype(str),sep=' '))

for (y,n) in zip(h, res.Names):

    ax.text(300,y+0.4, n, verticalalignment='center',color='white')

ax.set_xlabel('Popularity gain, babies per million');

ax.set_ylim(0,h.max()+1);