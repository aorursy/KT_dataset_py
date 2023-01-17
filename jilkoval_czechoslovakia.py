import pandas as pd

import numpy as np

data_all = pd.read_csv('../input/Indicators.csv')
# list of all indicators

indicators = data_all['IndicatorName'].unique().tolist()

print("There is", len(indicators), "indicators in the complete data set.")
filter_cz = data_all['CountryName'] == 'Czech Republic'

data_cz = data_all.loc[filter_cz]

print("Dataset for CZ has shape", data_cz.shape)



filter_sk = data_all['CountryName'] == 'Slovak Republic'

data_sk = data_all.loc[filter_sk]

print("Dataset for SK has shape", data_sk.shape)
# minimal number of data-points for the indicator to be copared

n_min_pre = 10

n_min_post = 10

n_min = n_min_pre + n_min_post
new_columns = ['corr_pre','corr_post','corr_diff','n_pre','n_post','IndicatorName']

corr_cs = pd.DataFrame()
for ind_i in indicators:

    # filter CZ and SK data for indicators

    f_cz = data_cz['IndicatorName'] == ind_i

    f_sk = data_sk['IndicatorName'] == ind_i

    

    if (sum(f_cz)>n_min) & (sum(f_sk)>n_min):

        # inner merge of data for CZ and SK

        ind_cz_sk = pd.merge(data_cz.loc[f_cz][['Year','Value']],

                             data_sk.loc[f_sk][['Year','Value']],

                             how='inner',

                             on='Year',

                             suffixes=('_cz', '_sk'))

        

        # filter data-points before and after 1993 (the separation happend on January 1st)

        f_pre = ind_cz_sk['Year']<1993

        f_post = 1993<=ind_cz_sk['Year']

        

        # calculate correlation for indicators where there is enough data before and after 1993

        if (sum(f_pre)>n_min_pre) & (sum(f_post)>n_min_post):

            corr_pre = (ind_cz_sk.loc[f_pre])['Value_cz'].corr((ind_cz_sk.loc[f_pre])['Value_sk'])

            corr_post = (ind_cz_sk.loc[f_post])['Value_cz'].corr((ind_cz_sk.loc[f_post])['Value_sk'])

            

            # check if the result is not NaN and append the result

            if ~np.isnan(corr_pre-corr_post):

                df_to_append = pd.DataFrame([[corr_pre, corr_post, corr_pre-corr_post, \

                                            sum(f_pre), sum(f_post), ind_i]], columns=new_columns)

                corr_cs = corr_cs.append(df_to_append, ignore_index=True)

                #print('{0:6.3f} {1:6.3f} {2:6.3f}'.format(corr_pre, corr_post, corr_pre-corr_post)+' '+\

                #      str(sum(f_pre))+' '+str(sum(f_post))+' '+ind_i)
print("Common indicators for CZ and SK before and after 1993, dataset shape:", corr_cs.shape)
# importing matplotlib first

%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt



# make font little larger

font = {'family' : 'sans-serif',

        'size' : 16}

matplotlib.rc('font', **font)
def plot_hist_corr(file_out=None):

    """

    plot histograms of the correlation coefficients derived for indicators

    before and after the division

    -- specify file_out for the plot in an output file

    """

    hist_line = {'histtype':'step',

                'alpha':0.7,

                'range':(-1,1),

                'lw':2.5}



    fig, ax = plt.subplots(figsize=(8, 6))

    plt.hist(corr_cs['corr_pre'],

             label='before division',

             **hist_line)

    plt.hist(corr_cs['corr_post'],

             ls='--',

             label='after division',

             **hist_line)



    plt.xlabel('correlation coefficient')

    plt.ylabel('indicators per bin')

    plt.legend(loc='upper left')

    plt.title('Distribution of correlation coefficients before and after 1993')

    

    if file_out is not None: plt.savefig(file_out)

    else: plt.show()
plot_hist_corr()
def plot_corr_scatter(file_out=None, title=None):

    """

    scatter plot with correlation coefficient before and after the division on the 

    horizontal and vertical axis, respectively.

    -- specify file_out for the plot in an output file

    """

    fig, ax = plt.subplots(figsize=(7, 7))

    

    plt.scatter(corr_cs['corr_pre'], corr_cs['corr_post'],

               alpha=0.6,

               s=100)

    

    plt.axis('equal')

    plt.xlabel('before division')

    ax.xaxis.set_label_coords(0.95,0.44)

    plt.ylabel('after division', rotation=0)

    ax.yaxis.set_label_coords(0.5,1.02)

    if title is None: plt.title('Correlation coefficients between CZ and SK\n before versus after 1993', y=1.1)

    

    plt.xlim(-1.1,1.1)

    plt.ylim(-1.1,1.1)

    ticks = [-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1]

    ax.set_xticks(ticks)

    ax.set_yticks(ticks)

    

    ax.spines['left'].set_position('zero')

    ax.spines['right'].set_color('none')

    ax.spines['bottom'].set_position('zero')

    ax.spines['top'].set_color('none')

    

    #ax.yaxis.tick_left()

    #ax.xaxis.tick_bottom()

    

    if file_out is not None: 

        plt.tight_layout()

        plt.savefig(file_out, dpi=400)

    else: plt.show()
plot_corr_scatter()
corr_cs = corr_cs.sort_values('corr_diff')

print(corr_cs.head())

print(corr_cs.tail())
# define colors and anmes for plotting of CZ and SK

colors = {'Czech Republic' : '#cb181d',

          'Slovak Republic' : '#41ab5d'}

names = {'Czech Republic' : 'CZ',

         'Slovak Republic' : 'SK'}



def plot_indicator_in_time(indicator_code, ylabel=None, title=None, plot_1993=True, file_out=None,

                          yscale=None):

    """

    Evolution of given indicator for CZ and SK.

    -- specify file_out for the plot in an output file

    """

    fig, ax = plt.subplots(figsize=(8, 6))

    

    for data_i in [data_cz, data_sk]:

        name_i = data_i['CountryName'].unique()[0]

        data_ind = data_i[data_i['IndicatorCode'] == indicator_code]

        if yscale is not None: val = data_ind['Value']/yscale

        else: val = data_ind['Value']

        plt.plot(data_ind['Year'], val,

                 color=colors[name_i],

                 label=names[name_i],

                 lw=3)

    

    ax.set_xlabel('Year')

    if ylabel is None: ylabel = data_ind['IndicatorName'].unique()[0]

    ax.set_ylabel(ylabel)

    

    if plot_1993:

        y_limits = ax.get_ylim()

        plt.plot([1993,1993],y_limits,

                lw=2,

                alpha=0.6,

                c='gray')

        plt.text(1993.5, 0.95*y_limits[1], '1993', color='gray', ha='left')

    

    if title is not None: ax.set_title(title)

    plt.legend()

    

    if file_out is not None: 

        plt.tight_layout()

        plt.savefig(file_out, dpi=400)

    else: plt.show()
# Population in largest city

plot_indicator_in_time('EN.URB.LCTY', 

                      yscale=1e5,

                      ylabel='Population in largest city [$10^5$]')
# Population in the largest city (% of urban population)

plot_indicator_in_time('EN.URB.LCTY.UR.ZS', 

                       ylabel='Population in the largest city\n(% of urban population)')
# CO2 emissions from transport (% of total fuel combustion)

plot_indicator_in_time('EN.CO2.TRAN.ZS',

                      ylabel='CO2 emissions from transport\n(% of total fuel combustion)')
# Death rate, crude (per 1,000 people)

plot_indicator_in_time('SP.DYN.CDRT.IN',

                      ylabel='Death rate, crude\n(per 1000 people)')