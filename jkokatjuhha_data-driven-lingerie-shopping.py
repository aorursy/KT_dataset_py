from IPython.display import HTML, display
html = "<table><tr><td>Hanky Panky: alexandra lace robe<img src='http://www.hankypankyltd.netdna-cdn.com/media/catalog/product/cache/1/image/db978388cfd007780066eaab38556cef/9/1/91r111_black_1.jpg' style='width: 900px;'></td><td>Macy's Calvin Klein<img src='https://cdna.lystit.com/200/250/tr/photos/saksfifthavenue/0400093782409-black-6d6c9f31-.jpeg' style='width: 900px;'></td></tr></table>"
display(HTML(html))
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import seaborn as sns
import numpy as np
# to normalize the strings
import unicodedata
import matplotlib.pyplot as plt
%matplotlib inline

pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_rows', 1000)

pals = sns.hls_palette(8, l=.3, s=.8)
purple = sns.cubehelix_palette(11)
light_purple = sns.light_palette("purple",n_colors=8)
pink = sns.color_palette("husl", 8)[7]
sns.set_palette(pals)
fig_size=[12,9]
sns.set(font_scale = 1.4)
sns.set_style('white', {'legend.frameon':True})
#sns.set_style("white")
plt.rcParams["figure.figsize"] = fig_size
brands_info = pd.read_csv('../input/cleaned-lingerie-data-from-different-brands/cleaned_data.csv')
brands_name = ['Calvin Klein','Hanky Panky',"Victoria's Secret", 'Topshop', 'Wacoal', 'Amazon Calvin Klein',
               'Amazon Hanky Panky', "Amazon Victoria's Secret", 'Amazon Wacoal',
               'Macys Calvin Klein', 'Macys Hanky Panky', 'Macys Wacoal']

df = brands_info.groupby('brand').product_name.agg(['count', 'nunique'])
df = df.loc[brands_name]
df.rename(columns={'count':'Number of products (colors)', 'nunique': 'Number of unique models' })
print(brands_info[brands_info['brand']=="Amazon Victoria's Secret"].product_name.iloc[1])
def manipul_regex(str_array):
    string = "|".join(str_array)
    return "(^|\s)(" + string + ")(\s|$)"

def categorize(data):
    panties = manipul_regex(["thong","g string","pant","v kini","boypant","pants","panty","thongs","panties",
                             "ladypant","knickers","thong","twist knicker","brief","boyshort",
                             "lace v front short","signature lace bike short","side tie bikini",
                             "signature lace string bikini","tanga","panty","hipster","vikini","b.sleek bikini",
                             "b.sultry bikini","b.sumptuous bikini","b.tempt'd lace kiss bikini",
                             "cheekster", 'boypants','ladypants', 'boyshorts', 'hiphugger', 
                             'pink high leg logo bikini', 'pink shortie', 'pink logo bikini',
                             'lace sexy shortie', 'body base shorty', 'b.splendid bikini'
                            ])  
    
    bodys = manipul_regex(["bodysuit","teddy","wink plaything"])
    bras = manipul_regex(["bra","bustier","strapless","balconette","bandeau",
                          'pink palm lace strappy push up','body by victoria unlined demi', 
                          'pink daisy lace racerback push up', 
                          'pink wildflower lace high neck push up',
                          'pink wear everywhere super push', 'pink seamless lightly lined racerback',
                          'body by wacoal seamless underwire','basic beauty wireless contour',
                          'pink lace lightly lined triangle', 'lace cross front unlined halter'
                            ])
    
    activewear = manipul_regex(['sports bra', 'sport bra', 'sport bralette', 'sports bralette'])
    suspenders = manipul_regex(["suspenders","belt"])
    bralettes = manipul_regex(["bralettes","bralette", 'bralet'])
    tops = manipul_regex(["tops","top"])
    babydoll = manipul_regex(["babydoll","camisole and bikini set by bluebella","chemise"])
    shorts = manipul_regex(["short","shorts","chiffon tap pant"])
    socks = manipul_regex(["socks"])
    pack = manipul_regex(["pack"])
    maternity = manipul_regex(["maternity"])
    basque = manipul_regex(["basque"])
    slip = manipul_regex(["slip"])
    bralette = manipul_regex(["bralette","bralettes"])
    garter = manipul_regex(["garter"])
    robe = manipul_regex(["robe", "kimonos","kimono"])
    camisole = manipul_regex(["camisole","cami","tank"])
    shapers = manipul_regex(["bike short","shaper","bodybriefer"])
    eyemask = manipul_regex(["eyemask"])
    hoodie = manipul_regex(["hoodie"])
    sleepwear = manipul_regex(["pajama","sleep set","sleepwear","modern cotton lounge pants"])
    rompers = manipul_regex(["romper"]) 
    
    dict_cats = [['hoodie',hoodie],['eyemask', eyemask],[ 'shapers',shapers],
                 [ 'garter', garter],['bralette',bralette],
                 ['slip',slip],[ 'basque',basque],['maternity',maternity],
                 ['pack',pack],['socks',socks],['shorts',shorts],['robe', robe],
                 ['tops',tops],
                 ['suspenders',suspenders],['rompers', rompers],
                 ['bras',bras],['panties',panties], ['babydoll', babydoll],
                 ['bodys',bodys],['sleepwear', sleepwear],['bralettes', bralettes], ['activewear', activewear]
                ]
    #because there is a babydool with g-string and teddy with thong

    
    data.loc[(data['brand'].str.contains('Macys')), 'product_category_gen'] = 'Other'
    data.loc[(data['brand']== "Victoria's Secret"), 'product_category_gen'] = 'Other'
    data.loc[(data['brand']== "Hanky Panky"), 'product_category_gen'] = 'Other'
    
    dict_cats_pro = [['camisoles',camisole]]
    for items in dict_cats_pro:
        naming, reg = items
        data.loc[(data['product_name'].str.contains(reg, case=False)), 'product_category_gen'] = naming
        data.loc[(data['description'].str.contains(reg, case=False)), 'product_category_gen'] = naming

    for items in dict_cats:
        naming, reg = items
        data.loc[(data['product_name'].str.contains(reg, case=False)), 'product_category_gen'] = naming
    
    #data.loc[(data['description'].str.contains('bralette|bralettes', case=False))&(data['product_name'].str.contains('bra')), 'product_category_gen'] = 'bralettes'

    #data.loc[(data['brand']== "Calvin Klein")&(data['product_name'].str.contains('hoodie', case=False)), 'product_category_gen'] = 'hoodie'        
    data.loc[(data['brand'].str.contains("Macys"))&(data['product_name'].str.contains('bikini', case=False)), 'product_category_gen'] = 'panties'
    data.loc[(data['brand']== "Victoria's Secret")&(data['product_name'].str.contains('dream angels new! lace-up pleated slip', case=False)), 'product_category_gen'] = 'babydoll'
    data.loc[(data['brand']== "Victoria's Secret")&(data['product_name'].str.contains('very sexy lace plunge garter slip', case=False)), 'product_category_gen'] = 'babydoll'
    #data.loc[(data['brand']== "Victoria's Secret")&(data['pdp_url'].str.contains('swim', case=False)), 'product_category_gen'] = 'swimsuits'
    data.loc[(data['descr_conc'].str.contains('bralette|bralettes'))&(data['product_name'].str.contains('bandeau', case=False)), 'product_category_gen'] = 'bralettes'

    data.loc[(data['brand']== "Victoria's Secret")&(data['pdp_url'].str.contains('activewear', case=False)), 'product_category_gen'] = 'activewear'
    data.loc[(data['pdp_url'].str.contains('swim', case=False)), 'product_category_gen'] = 'swimsuits'
    data.loc[(data['descr_conc'].str.contains('swim', case=False)), 'product_category_gen'] = 'swimsuits'

    data.loc[(data['brand']== "Calvin Klein")&(data['pdp_url'].str.contains('activewear', case=False)), 'product_category_gen'] = 'activewear'

    data['product_category_gen'] = data['product_category_gen'].apply(lambda x: x.replace("women - lingerie & shapewear - ", ""))

    data.loc[(data['brand']== "Topshop")&(data['product_name'].str.contains('calvin klein', case=False)), 'product_category_gen'] = 'calvin klein'
    data.loc[(data['brand']== "Topshop")&(data['pdp_url'].str.contains('body', case=False)), 'product_category_gen'] = 'bodys'


    data.loc[(data['brand']== "Hanky Panky")&(data['product_name'].str.contains(manipul_regex(['top|crop']), case=False)), 'product_category_gen'] = 'bras'
    data.loc[(data['brand']== "Hanky Panky")&(data['product_name'].str.contains('bikini', case=False)), 'product_category_gen'] = 'panties'
    data.loc[(data['brand']== "Hanky Panky")&(data['product_name'].str.contains('slip', case=False)), 'product_category_gen'] = 'slip'
    data.loc[(data['brand']== "Hanky Panky")&(data['product_name'].str.contains('t-shirt', case=False)), 'product_category_gen'] = 'camisoles'
    #data.loc[(data['brand']== "Hanky Panky")&(data['descr_conc'].str.contains('panty', case=False)), 'product_category_gen'] = 'panties'


    data.loc[(data['brand']== "Hanky Panky")&(data['product_name'].str.contains('low rise thong rose bouquet of|original rise thong rose bouquet of', case=False)), 'product_category_gen'] = 'pack'
    #data.loc[(data['brand']== "Hanky Panky")&(data['product_name'].str.contains('low rise thong rose bouquet of 12', case=False)), 'product_category_gen'] = 'pack'
    #data.loc[(data['brand']== "Hanky Panky")&(data['product_name'].str.contains('original rise thong rose bouquet of 6', case=False)), 'product_category_gen'] = 'pack'
    #data.loc[(data['brand']== "Hanky Panky")&(data['product_name'].str.contains('original rise thong rose bouquet of 12', case=False)), 'product_category_gen'] = 'pack'
    data.loc[(data['product_name'].str.contains(manipul_regex(['pack', 'kiss packaged']), case=False)), 'product_category_gen'] = 'pack'

    #data.loc[(data['product_category'].str.contains('panty sets', case=False)), 'product_category_gen'] = 'panty sets'
    #data.loc[(data['brand']== "Victoria's Secret")&(data['product_category'].str.contains('cami & short set', case=False)), 'product_category_gen'] = 'sleep shorts & top'

    #['widow and thong', 'sultry balconette bra, thong and garter belt']
    #bra_panties_set = ['widow & thong', 'bahama mama lace',
    #                   'bra &','b.sultry balconette bra, thong and garter belt',
    #                   '^(?=.*bra)(?=.*(bikini|panty|tanga|hipster|thong|short|trunk|high cut brief|knickers))']
    #data.loc[(data['product_category'].str.contains('lingerie sets', case=False)), 'product_category_gen'] = 'bra panties sets'
    data.loc[(data['descr_conc'].str.contains('widow and thong', case=False)), 'product_category_gen'] = 'bra panties sets'


    data.loc[(data['product_name'].str.contains('sleeve top', case=False)), 'product_category_gen'] = 'tops'
    data.loc[(data['product_name'].str.contains('sleep set', case=False)), 'product_category_gen'] = 'sleepwear'
    data.loc[(data['product_category'].str.contains('sleepwear'))&(data['pdp_url'].str.contains('sleepwear', case=False)), 'product_category_gen'] = 'sleepwear'
    data.loc[(data['brand']=="Victoria's Secret")&(data['product_name'].isin(['pink wear everywhere t shirt bra','pink lace t shirt bra'])), 'product_category_gen'] = 'bras'
    data.loc[(data['brand']=="Macys Wacoal")&(data['product_name'].str.contains('embrace lace soft cup wireless bra')), 'product_category_gen'] = 'bras'
    data.loc[(data['brand']=="Victoria's Secret")&(data['product_name']=='pink super soft beach cheeky cropped tank'), 'product_category_gen'] = 'camisoles'
    # we will assume that there is no such thing as sport underwear
    data.loc[(data['product_name'].str.contains(bras))&(data['brand']== "Calvin Klein")&(data['pdp_url'].str.contains('sports-bras', case=False)), 'product_category_gen'] = 'activewear'
    data.loc[(data['product_name'].str.contains(bralettes))&(data['brand']== "Calvin Klein")&(data['pdp_url'].str.contains('sports-bras', case=False)), 'product_category_gen'] = 'activewear'
    data.loc[(data['brand']=='Topshop')&(data['product_name'].isin(['topshop branded sporty bralet','seamless sporty branded bra','seamless sporty bra', 'sporty bra','sporty branded bra'])), 'product_category_gen'] = 'activewear'
    data.loc[(data['brand']=='Macys Wacoal')&(data['product_name'].str.contains('sport high impact underwire bra')), 'product_category_gen'] = 'activewear'
    data.loc[(data['brand']=="Victoria's Secret")&(data['product_name'].isin(['body by victoria daisy lace sleep tee','macrame lace mix sleep short','double v sleep romper'])), 'product_category_gen'] = 'sleepwear'
    data.loc[(data['brand']=="Hanky Panky")&(data['product_name'].isin(['violet spray chiffon sleep top'])), 'product_category_gen'] = 'sleepwear'

    data.loc[(data['pdp_url']=='https://www.amazon.com/-/dp/b06wrtg5d8?th=1&psc=1'), 'product_category_gen'] = 'bra panties sets'
    data.loc[(data['product_name'].str.contains('maternity')), 'product_category_gen'] = 'maternity'

    return data
brands_info = categorize(brands_info)
brands_info['product_category_gen'].unique()
display(brands_info[(brands_info['product_category_gen']=='lingerie')])
display(brands_info[(brands_info['pdp_url'].str.contains('sleep')) & (brands_info['brand']=="Victoria's Secret")].product_name.unique())
for name, group in brands_info.groupby('brand'):
    print(name)
    
    df = group.drop_duplicates(['product_name'],inplace=False)
    df = pd.DataFrame(df.groupby('product_category_gen').size()).sort_values(by = 0, ascending=False).transpose()
    df_per = df.apply(lambda r: r/r.sum() * 100, axis=1)
    df = df_per.append(df)
    df.index = ['percent (m)', 'absolut (m)']
    
    
    tab = pd.DataFrame(group.groupby('product_category_gen').size()).sort_values(by = 0, ascending=False).transpose()
    cols = tab.columns.tolist()
    tab_per = tab.apply(lambda r: r/r.sum() * 100, axis=1)
    tab = tab_per.append(tab)
    tab.index = ['percent (m+c)', 'absolut (m+c)']
    
    display(tab.append(df)[cols])
ind = brands_info[brands_info['product_category_gen']=='calvin klein'].index
brands_info.drop(ind, inplace = True)
categories=['bras','panties', 'bralettes', 'bodys', 'swimsuits', 'activewear','sleepwear', 'pack']
df = brands_info[brands_info['product_category_gen'].isin(categories)]
df = df.groupby(['product_category_gen', 'brand']).product_name.nunique()
df = pd.DataFrame(df)
df = df['product_name'].groupby(level=0, group_keys=False)
display(df.nlargest(6))
del df
bras = manipul_regex(["bra","bustier","strapless","balconette","bandeau"])
bralettes = manipul_regex(["bralettes","bralette", 'bralet'])
brands_info[(~brands_info['product_name'].str.contains(bralettes))&(brands_info['product_name'].str.contains(bras))&(brands_info['brand']=="Victoria's Secret")].product_name.nunique()    
categories=['bras','panties', 'bralettes', 'bodys', 'swimsuits', 'sleepwear', 'babydoll', 'pack']
df = brands_info[brands_info['product_category_gen'].isin(categories)]
df = df.groupby(['product_category_gen', 'brand']).product_name.agg(['size'])
df = pd.DataFrame(df)
df = df['size'].groupby(level=0, group_keys=False)
#display(df.nlargest(6))
df = pd.DataFrame(df.nlargest(6))
display(df)
del df
#df.to_csv('category_brand.csv')
display(brands_info.groupby('brand').color.nunique())
(brands_info[brands_info['brand']=="Victoria's Secret"].groupby('color').size()).head(20)
colors_df = pd.DataFrame()    
for name, group in brands_info.groupby('brand'):
    colors = group.groupby(['product_name']).color.nunique()
    colors_df = colors_df.append(pd.DataFrame({'color_num': colors, 'brand': name}))

fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(20,8))
ax = sns.boxplot(x='color_num', y='brand', data=colors_df, ax=axes, medianprops={"zorder":3})
ax = sns.swarmplot(x="color_num", y="brand", data=colors_df, color='grey', alpha=0.7)
ax.xaxis.grid(True)
ax.set_xlabel('Number of colors per product', fontsize=17);
ax.tick_params(axis='both', which='major', labelsize=17);

colors_df.groupby('brand').agg(['median', 'mean', 'count']).transpose()
display(colors_df[colors_df['color_num']>=35])
del colors_df
#brands_info[brands_info['product_name'] == 'victoria sport the player racerback sport bra by victoria sport'].color.unique()
colors_first = brands_info.groupby(['brand','color'])['color'].agg({'count_fir':'count'})
df_count_first = colors_first.groupby(level=0).agg('idxmax')
df_count_first = colors_first.loc[df_count_first['count_fir']].reset_index()

data_mod = brands_info[~brands_info['color'].isin(df_count_first.color.unique())]
colors_second = data_mod.groupby(['brand','color'])['color'].agg({'count_sec':'count'})
df_count_second = colors_second.groupby(level=0).agg('idxmax')
df_count_second = colors_second.loc[df_count_second['count_sec']].reset_index()

df = df_count_first.merge(df_count_second, on='brand')
df.columns = ['brand', '1st color', 'count 1st color', '2nd color', 'count 2nd color']
display(df)

for x in [df_count_first, df_count_second, df, data_mod, colors_first]:
    del x
main_categories = ['bralettes','activewear','bras','panties','bodys','pack','babydoll', 'swimsuits', 'sleepwear']
brands_info['product_category'] = brands_info['product_category_gen']
brands_info['category'] = brands_info['product_category_gen']
brands_info.loc[~brands_info['category'].isin(main_categories), 'category'] = 'Other'
del brands_info['product_category_gen']
fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(20,5))
ax = sns.boxplot(x='mrp', y= 'brand', data = brands_info, palette = purple, order = brands_name) 
brands_info[(brands_info['mrp']>300)&(brands_info['brand']=='Hanky Panky')].product_name.unique()
def general_plot(money = 'mrp'):
    fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(20,5))
    ax = sns.boxplot(x=money, y= 'brand', data = brands_info[brands_info[money]<150], palette = purple, order = brands_name) 
    _ = ax.set_xticklabels(ax.get_xticklabels(), fontsize=13)


    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(20,4))
    ax.xaxis.grid(True)
    ax = sns.barplot(y=money, x = 'brand', data = brands_info, palette = light_purple[1:], ax = axes[0], order = brands_name)
    ax.yaxis.grid(True)
    _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=13)
    ax = sns.barplot(y=money, x = 'brand', data = brands_info, palette = light_purple[1:], estimator=np.median, ax = axes[1], order = brands_name)
    ax.yaxis.grid(True)
    _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=13)
general_plot()
general_plot(money = 'price')
brands_info[(brands_info['mrp']>120) & (brands_info['category']!='pack')].product_category.unique()
brands_info[brands_info['product_name']=='alexandra lace robe']
from IPython.display import HTML, display
html = "<table><tr><td>Hanky Panky: sophia lace bodysuit<img src='http://www.hankypankyltd.netdna-cdn.com/media/catalog/product/cache/1/image/db978388cfd007780066eaab38556cef/7/9/798504_black_4.jpg' style='width: 800px;'></td><td>Hanky Panky: short european flannel robe<img src='http://www.hankypankyltd.netdna-cdn.com/media/catalog/product/cache/1/image/db978388cfd007780066eaab38556cef/2/v/2vr114_1.jpg' style='width: 800px;'></td><td>Hanky Panky: alexandra lace robe<img src='http://www.hankypankyltd.netdna-cdn.com/media/catalog/product/cache/1/image/db978388cfd007780066eaab38556cef/9/1/91r111_black_1.jpg' style='width: 900px;'></td><td>Hanky Panky: wink plaything<img src='http://www.hankypankyltd.netdna-cdn.com/media/catalog/product/cache/1/image/db978388cfd007780066eaab38556cef/4/t/4t8504_black_1.jpg' style='width: 900px;'></td><td>Hanky Panky: victoria chemise with g string<img src='http://www.hankypankyltd.netdna-cdn.com/media/catalog/product/cache/1/image/db978388cfd007780066eaab38556cef/9/4/945901_2.jpg' style='width: 900px;'></td><td>Hanky Panky: heather jersey kimono robe<img src='http://www.hankypankyltd.netdna-cdn.com/media/catalog/product/cache/1/image/db978388cfd007780066eaab38556cef/6/8/68r222_wine_1.jpg' style='width: 900px;'></td><td>Macy's Calvin Klein<img src='https://cdna.lystit.com/200/250/tr/photos/saksfifthavenue/0400093782409-black-6d6c9f31-.jpeg' style='width: 900px;'></td></tr></table>"
display(HTML(html))
df = pd.DataFrame(brands_info.groupby('brand').mrp.max()).reset_index()
df2 = pd.DataFrame(brands_info.groupby('brand').price.max()).reset_index()
df = df.merge(df2, on='brand')
display(df)
expensive_items_mrp = pd.DataFrame()
expensive_items_price = pd.DataFrame()

for ind, row in df.iterrows():
    df_temp = brands_info[(brands_info['brand']==row['brand'])&(brands_info['mrp']==row['mrp'])]
    names = df_temp.product_name.unique()
    expensive_items_mrp = expensive_items_mrp.append(df_temp.iloc[0])

    df_temp = brands_info[(brands_info['brand']==row['brand'])&(brands_info['price']==row['price'])]
    listings = list(set(df_temp.product_name) - set(names))
    if len(listings)!=0:
        df_temp = df_temp[(df_temp['product_name'].isin(listings))]
        expensive_items_price = expensive_items_price.append(df_temp.iloc[0])
#display(expensive_items_mrp.head(5))

html = "<table><tr><td>Calvin Klein<img src='https://cdnd.lystit.com/1040/1300/n/photos/shopbop/CKLEN4233210437-Black-04ad40dc-.jpeg' style='width: 900px;'></td><td>Hanky Panky<img src='http://www.hankypankyltd.netdna-cdn.com/media/catalog/product/cache/1/image/db978388cfd007780066eaab38556cef/4/8/48pk30_emul_1_1.jpg' style='width: 800px;'></td><td>Victorias Secret<img src='https://dm.victoriassecret.com/p/760x1013/tif/b4/4f/b44f5a12e28c4b8c9dcab27faae01013/V493406_OF_F.jpg' style='width: 900px;'></td><td>Topshop<img src='http://media.topshop.com/wcsstore/TopShopDE/images/catalog/TS43E04LLAC_Zoom_F_1.jpg' style='width: 900px;'></td><td>Wacoal<img src='https://a248.e.akamai.net/f/248/9086/10h/origin-d5.scene7.com/is/image/WacoalAmerica/953237_004?qlt=85,1&op_sharpen=1&resMode=bilin&op_usm=1,1,9,0&rect=0,0,261,344&scl=5.816860465116279&id=VFprq1' style='width: 900px;'></td></tr></table>"
display(HTML(html))
html = "<table><tr><td>Amazon Calvin Klein<img src='https://images-na.ssl-images-amazon.com/images/I/61wmAanrDzL._UL1500_.jpg' style='width: 900px;'></td><td>Amazon Victorias Secret<img src='https://images-na.ssl-images-amazon.com/images/I/71CxBauOwPL._UL1500_.jpg' style='width: 900px;'></td><td>Amazon Hanky Panky<img src='https://images-na.ssl-images-amazon.com/images/I/71gd1OukSaL._UL1500_.jpg' style='width: 900px;'></td><td>Amazon Wacoal<img src='https://images-na.ssl-images-amazon.com/images/I/81uAK8Lvx8L._UL1500_.jpg' style='width: 900px;'></td><td>Macys Calvin Klein<img src='https://cdna.lystit.com/200/250/tr/photos/saksfifthavenue/0400093782409-black-6d6c9f31-.jpeg' style='width: 900px;'></td><td>Macys Hanky Panky<img src='https://image.shopittome.com/apparel_images/fb/hanky-panky-hanky-panky-eyelash-lace-bodysuit-4t8504-abv6af8ff20_zoom.jpg' style='width: 900px;'></td><td>Macys Wacoal<img src='https://slimages.macysassets.com/is/image/MCY/products/0/optimized/3831480_fpx.tif?op_sharpen=1&wid=400&hei=489&fit=fit,1&$filterlrg$' style='width: 900px;'></td></tr></table>"
display(HTML(html))
#display(expensive_items_price)
html = "<table><tr><td>Amazon Calvin Klein<img src='https://images-na.ssl-images-amazon.com/images/I/81LEl1kn5oL._UL1500_.jpg' style='width: 200px;'></td><td>Macys Wacoal<img src='https://slimages.macysassets.com/is/image/MCY/products/5/optimized/3640215_fpx.tif??op_sharpen=1&fit=fit,1&$filterlrg$&wid=1200&hei=1467' style='width: 200px;'></td><td>Wacoal<img src='https://images.prod.meredith.com/product/96a181a21f45ac2c2183d2076f789d9d/53fc48ce1c2d384f512eaa489f2011980c2dc667471d96e0ad01570626b111b8/l/womens-b-temptd-by-wacoal-b-sultry-chemise' style='width: 200px;'></td></tr></table>"
display(HTML(html))
def create_plots(cols, swarm = False, inner = None, money = 'mrp'):
    df = brands_info[brands_info[money]<=100]
    df2 = df[df['brand'].isin(cols)]
    fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(15,5))
    ax = sns.violinplot(y=money, x='brand', data=df2, split=True,inner=inner, ax = axes, palette = light_purple)
    _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=90,fontsize=13)
    if swarm:
        ax = sns.swarmplot(x="brand", y=money, data=df2, color="black", alpha=.5, ax = axes)
    ax.yaxis.grid(True)
    fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(20,6))
    ax = sns.boxplot(y=money,hue='category',x = 'brand',data = df2, palette=sns.color_palette("husl", 10))
    _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=90,fontsize=13)
    ax.yaxis.grid(True)
    _ = ax.set_ylabel(money)    
create_plots(['Calvin Klein', "Victoria's Secret", 'Topshop', 'Wacoal', 'Hanky Panky', 'Amazon Calvin Klein', 'Macys Calvin Klein','Amazon Hanky Panky', 'Macys Hanky Panky','Amazon Wacoal', 'Macys Wacoal'])
create_plots(['Calvin Klein', "Victoria's Secret", 'Topshop', 'Wacoal', 'Hanky Panky'])
create_plots(['Calvin Klein', 'Amazon Calvin Klein', 'Macys Calvin Klein'], swarm = True)
df = brands_info[(brands_info['brand'].isin(['Amazon Calvin Klein','Calvin Klein']))]
fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(14,5))
ax = sns.violinplot(y="mrp", hue ='brand', x = 'product_category', split=True, data=df, ax = axes, palette = [light_purple[2], light_purple[5]])
ax.yaxis.grid(True)
def find_price_per_panty(brand):
    
    pack_panties = brands_info[(brands_info['brand']==brand)&(brands_info['category']=='pack')]
    pack_panties = pack_panties[~pack_panties['product_name'].str.contains(manipul_regex(['bra', 'bralette']))]
    def find_digit(x):
        for s in x.split():
            if s.isdigit():
                return int(s)
    pack_panties['panties_num'] = pack_panties['product_name'].apply(lambda x: find_digit(x))
    mrp = (pack_panties['mrp']/pack_panties['panties_num'])
    price = (pack_panties['price']/pack_panties['panties_num'])
    return [mrp,price]
hankypanky_mrp, hankypanky_price = find_price_per_panty('Hanky Panky')
macysck_mrp, macysck_price = find_price_per_panty('Macys Calvin Klein')
amazonck_mrp, amazonck_price = find_price_per_panty('Amazon Calvin Klein')

packs_price = pd.DataFrame({'brand': 'Amazon Calvin Klein', 'mrp': amazonck_mrp})
packs_price = packs_price.append(pd.DataFrame({'brand': 'Macys Calvin Klein', 'mrp': macysck_mrp})) 
packs_price = packs_price.append(pd.DataFrame({'brand': 'Hanky Panky', 'mrp': hankypanky_mrp})) 
fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(10,2))
ax = sns.boxplot(y='brand', x='mrp',data = packs_price, ax = axes, medianprops={"zorder":3},
                palette = [light_purple[2],light_purple[6]])
ax = sns.swarmplot(y='brand', x='mrp',data = packs_price, color = 'grey',alpha=0.7)
packs_price = pd.DataFrame({'brand': 'Amazon Calvin Klein', 'price': amazonck_price})
packs_price = packs_price.append(pd.DataFrame({'brand': 'Macys Calvin Klein', 'price': macysck_price})) 
packs_price = packs_price.append(pd.DataFrame({'brand': 'Hanky Panky', 'price': hankypanky_price})) 

fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(8,2))
ax = sns.boxplot(y='brand', x='price',data = packs_price, ax = axes, medianprops={"zorder":3},
                 palette = [light_purple[2],light_purple[6]])
ax = sns.swarmplot(y='brand', x='price',data = packs_price, color = 'grey',alpha=0.7)
ax.set_xlabel('price per panty from a pack')
packs_price = pd.DataFrame({'brand': 'Amazon Calvin Klein', 'mrp': amazonck_mrp})
packs_price = packs_price.append(pd.DataFrame({'brand': 'Macys Calvin Klein', 'mrp': macysck_mrp})) 

fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(10,2))
ax = sns.boxplot(y='brand', x='mrp',data = packs_price, ax = axes, medianprops={"zorder":3},
                palette = [light_purple[2],light_purple[6]])
ax = sns.swarmplot(y='brand', x='mrp',data = packs_price, color = 'grey',alpha=0.7)
ax.set_xlabel('price per panty from a pack',fontsize=13)
_= ax.tick_params(axis='both', which='major', labelsize=13)
create_plots(['Wacoal', 'Amazon Wacoal', 'Macys Wacoal'], swarm = True)
create_plots(['Hanky Panky', 'Amazon Hanky Panky', 'Macys Hanky Panky'],swarm = True)
def create_plots(cols, swarm = False, inner = None, money = 'mrp'):
    df = brands_info[brands_info[money]<=100]
    df2 = df[df['brand'].isin(cols)]
    fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(15,5))
    ax = sns.violinplot(x=money, y='brand', data=df2, split=True,inner=inner, ax = axes, palette = [light_purple[0], light_purple[2], light_purple[5]])
    if swarm:
        ax = sns.swarmplot(x="brand", y=money, data=df2, color="black", alpha=.5, ax = axes)
    ax.yaxis.grid(True)
    fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(20,5))
    ax = sns.boxplot(y=money,hue='category',x = 'brand',data = df2, palette=sns.color_palette("husl", 9))
    ax.yaxis.grid(True)
    _ = ax.set_ylabel(money)
create_plots(["Victoria's Secret", "Amazon Victoria's Secret"],swarm = False)
def category_plots(category, data, y= 'brand',xtitle = 'mrp',money = 'mrp', swarm = True, 
                   figsize1=(15,6), figsize2=(15,6), figsize3=(20,4),
                  filename = None):
    df = data[data['category'].isin(category)]
    fig, axes = plt.subplots(nrows=1, ncols=1,figsize=figsize1)
    #ax = sns.boxplot(x=money, hue ='brand', y = 'category',  data=df, ax = axes)
    ax = sns.boxplot(x=money, y= y, data=df, ax=axes, medianprops={"zorder":3})
    if swarm:
        ax = sns.swarmplot(y = y, x=money, data=df, ax = axes, color = 'grey', alpha=0.7)
    ax.xaxis.grid(True)
    ax.set(xlabel=xtitle)
    
    fig, axes = plt.subplots(nrows=1, ncols=1,figsize=figsize2)
    ax = sns.violinplot(y=money, x=y, data=df, ax=axes, cut = 0)
    _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.yaxis.grid(True)
    ax.set(ylabel=xtitle)
    _ = ax.set_title(", ".join(category))
    if filename!= None:
        fig.savefig(filename + '1.png')
    barplots(money,df, figsize3, x=y, xtitle = xtitle)
    


def barplots(money, df, figsize3, xtitle = 'mrp', x = 'brand', hue = None, palette = light_purple[1:]):
    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=figsize3)
    ax = sns.barplot(y=money, x = x, hue = hue, data = df, palette = palette, ax = axes[0])
    ax.set(ylabel='mean(' + xtitle + ')')
    ax.yaxis.grid(True)
    _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax = sns.barplot(y=money, x = x, hue = hue, data = df, palette = palette, estimator=np.median, ax = axes[1])
    ax.set(ylabel='median(' + xtitle + ')')
    display(df.groupby('brand')[money].agg({'median'}).transpose())
    display(df.groupby('brand')[money].agg({'mean'}).transpose())

    ax.yaxis.grid(True)
    _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

for name, group in brands_info.groupby('brand'):
    print(name)
    print((group[(group['category']=='bras')].mrp<40).sum())
category_plots(['bras'], brands_info,
               figsize1=(15,5), figsize2=(12,4), figsize3=(12,3), filename = 'bras_mrp')
category_plots(['panties'], brands_info, swarm = False, figsize1=(12,5),figsize2=(20,5), figsize3=(15,3))
category_plots(['bralettes'], brands_info, figsize1=(20,5),figsize2=(20,5), figsize3=(15,3))
category_plots(['bodys'], brands_info, figsize1=(15,3), figsize2=(10,3), figsize3=(10,2))
category_plots(['activewear'], brands_info, figsize1=(15,4), figsize2=(10,3), figsize3=(10,2))
category_plots(['sleepwear'], brands_info, figsize1=(15,2), figsize2=(10,3), figsize3=(10,2))
def barplots_extra(money, df, figsize3, xtitle = 'mrp', x = 'brand', hue = None, palette = light_purple[1:]):
    fig, axes = plt.subplots(nrows=2, ncols=1,figsize=figsize3)
    plt.subplots_adjust(hspace = 0.5, wspace=0.5)
    ax = sns.barplot(y=money, x = x, hue = hue, data = df, palette = palette, ax = axes[0])
    ax.set_ylabel('mean(' + xtitle + ')', fontsize=13)
    ax.set_xlabel('category', fontsize=12)
    ax.yaxis.grid(True)
    _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    _= ax.tick_params(axis='both', which='major', labelsize=14)

    ax = sns.barplot(y=money, x = x, hue = hue, data = df, palette = palette, estimator=np.median, ax = axes[1])
    ax.set_ylabel('median(' + xtitle + ')', fontsize=13)
    display(df.groupby('brand')[money].agg({'median'}).transpose())
    ax.set_xlabel('category', fontsize=12)
    _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    _= ax.tick_params(axis='both', which='major', labelsize=14)
    ax.yaxis.grid(True)

sns.set(font_scale = 1.2)
sns.set_style('white', {'legend.frameon':True})
df = brands_info[brands_info['brand'].isin(['Calvin Klein', "Victoria's Secret"])]
df = df[df['product_category']!='Other']
fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(15,6))
ax = sns.boxplot(y = 'category', x="mrp", hue='brand', data=df, ax=axes, palette = [light_purple[2], light_purple[5]])
#ax = sns.swarmplot(y = 'category', hue = 'brand', x="mrp", data=df, ax = axes, color = 'grey')
_= ax.tick_params(axis='both', which='major', labelsize=13)
ax.set_xlabel('mrp', fontsize=13)
ax.set_ylabel('category', fontsize=13)
ax.xaxis.grid(True)
fig.savefig('calvin_victoria_boxplot.png')

fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(15,6))
ax = sns.violinplot(y="mrp", hue ='brand', x = 'category',cut = 0, split=True, data=df, ax = axes, palette = [light_purple[2], light_purple[5]])
ax.yaxis.grid(True)
_= ax.tick_params(axis='both', which='major', labelsize=13)
ax.set_ylabel('mrp', fontsize=13)
ax.set_xlabel('category', fontsize=13)
fig.savefig('calvin_victoria_violin.png')

barplots_extra('mrp', df, (6,10), hue = 'brand', x='category', palette = [light_purple[2], light_purple[5]])

def find_discount(data):
    data['diff'] = (data['mrp'] - data['price']).round(2)
    data['diff_categories'] = ''
    data.loc[(data['diff'] < 9) & (data['diff'] != 0.0), 'diff_categories'] = 'low'
    data.loc[(data['diff'] >= 9) & (data['diff'] < 18), 'diff_categories'] = 'middle'
    data.loc[(data['diff'] >= 18), 'diff_categories'] = 'high'
    data.loc[(data['diff'] == 0.0), 'diff_categories'] = 'no'   
    data['discount'] = "not discounted"
    data.loc[(data['diff'] > 0.0), 'discount'] = 'discounted'   
    return data
brands_info = find_discount(brands_info)
def stacked_all(data, category, name = None, discount_cat = 'diff_categories'):
    fig, ax = plt.subplots(nrows=2, ncols=1,figsize=(15,10))
    tab = pd.crosstab(data[discount_cat], data[category]).transpose()
    if 'high (>=50%)' in tab.columns:
        ordering = ['high (>=50%)','middle (20%>=x<50%)','low (<20%)','no']
    else:
        ordering = ['high','middle','low','no']
    cor = [x for x in ordering if x in tab.columns] + [x for x in tab.columns if x not in ordering]
    tab = tab[cor]

    tab_prop = tab.div(tab.sum(1).astype(float), axis=0)  
    tab_pro = tab_prop.append(pd.DataFrame([1])).drop(0, axis=1) 
    display(tab_pro.transpose())
    ##f7caf2
    colors = ["#755791","#c06c84", "#f67280", "#f8b195"] 
    

    tab_pro.reindex(tab_pro.index.tolist()).plot(kind="bar", stacked=True, ax = ax[0], fontsize = 9, color = colors)  
    ax[0].yaxis.grid(True)
    df = tab.transpose()
    bbox=[0, 0, 1, 1]
    ax[1].axis('off')
    mpl_table = ax[1].table(cellText = df.values, rowLabels = df.index, 
                            colLabels=df.columns, bbox = bbox)

    plt.legend(fontsize=9)
    #ax[0].set_xlabel("Difference between RMP and price", fontsize = 10 )
    ax[0].set_title(name)

def stacked_all(data, category, name = None):
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(6,3))
    tab = pd.crosstab(data['diff_categories'], data[category]).transpose()
    print(name)
    display(tab)

    tab_prop = tab.div(tab.sum(1).astype(float), axis=0)  
    tab_pro = tab_prop.append(pd.DataFrame([1])).drop(0, axis=1) 
    display(tab_pro.transpose())

    tab_pro.reindex(tab_pro.index.tolist()).plot(kind="bar", stacked=True, ax = ax, fontsize = 9)

stacked_all(brands_info, 'category')
tab = pd.crosstab(brands_info['discount'], brands_info['brand']).transpose()
display(tab.transpose())

tab_prop = tab.div(tab.sum(1).astype(float), axis=0)  
tab_pro = tab_prop.append(pd.DataFrame([1])).drop(0,axis=1) 
tab_pro.drop(tab_pro.index[12], inplace = True)
display(tab_pro.transpose())
colors = ["#755791", "#f67280"]
fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(8,6))
_ = tab_pro.reindex(tab_pro.index.tolist()).plot(kind="bar", stacked=True, ax = ax, fontsize = 12, color = colors)
_= ax.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
fig.savefig('percent_discount.png')

def stacked_all_brands(data, discount_cat = 'diff_categories', figsize = (15,5)):
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=figsize)
    tab = pd.crosstab(data[discount_cat], data['brand']).transpose()
    if 'high (>=50%)' in tab.columns:
        ordering = ['high (>=50%)','middle (20%>=x<50%)','low (<20%)','no']
    else:
        ordering = ['high','middle','low','no']
    cor = [x for x in ordering if x in tab.columns] + [x for x in tab.columns if x not in ordering]
    tab = tab[cor]
    display(tab.transpose())

    tab_prop = tab.div(tab.sum(1).astype(float), axis=0)  
    tab_pro = tab_prop.append(pd.DataFrame([1])).drop(0, axis=1) 
    display(tab_pro.transpose())
    ##f7caf2
    colors = ["#755791","#c06c84", "#f67280", "#f8b195"] 
    tab_pro.reindex(tab_pro.index.tolist()).plot(kind="bar", stacked=True, ax = ax, fontsize = 9, color = colors)
    ax.yaxis.grid(True)
df = brands_info[brands_info['diff']>0]
category_plots(brands_info['category'].unique(), df, swarm = True, money = 'diff', xtitle = 'difference')
brands_info[brands_info['diff']>40][['product_name','brand','category','diff', 'price', 'mrp', 'color']]
html = "<table><tr><td>Victorias Secret: bra<img src='https://images-na.ssl-images-amazon.com/images/I/713DKKJk4dL._UY445_.jpg' style='width: 200px;'></td><td>Victorias Secret: slip<img src='https://images-na.ssl-images-amazon.com/images/I/61CWtyQDgjL._UL1500_.jpg' style='width: 200px;'></td><td>Topshop: bra by bluebella<img src='http://media.topshop.com/wcsstore/TopShopDE/images/catalog/TS43C62LBLK_Zoom_F_1.jpg' style='width: 200px;'></td></tr></table>"
display(HTML(html))
html = "<table><tr><td>Amazon Calvin Klein: least discounted panty<img src='https://images-na.ssl-images-amazon.com/images/I/81AmIezUmWL._UL1500_.jpg' style='width: 200px;'></td></tr></table>"
display(HTML(html))
min_diff = np.min(brands_info[brands_info['diff']>0]['diff'])
brands_info[brands_info['diff']==min_diff]
def find_percentage_discount(data):
    data['diff_per'] = (data['diff'] * 100  / data['mrp']).round(2)  

    data['diff_per_categories'] = ''
    data.loc[(data['diff_per'] < 20) & (data['diff_per'] != 0.0), 'diff_per_categories'] = 'low (<20%)'
    data.loc[(data['diff_per'] >= 20) & (data['diff_per'] < 50), 'diff_per_categories'] = 'middle (20%>=x<50%)'
    data.loc[(data['diff_per'] >= 50), 'diff_per_categories'] = 'high (>=50%)'
    data.loc[(data['diff_per'] == 0.0), 'diff_per_categories'] = 'no'   
    return data
brands_info = find_percentage_discount(brands_info)
df = brands_info
df = df[df['diff_per']>0]
category_plots(df['category'].unique(), df, swarm = True, money = 'diff_per', xtitle='% discount',
               figsize1=(15,5), figsize2=(15,4), figsize3=(10,2))
df[df['diff_per']>50].groupby('brand').size()
df = brands_info[brands_info['diff_per']>=75]
df.drop_duplicates(['product_name','price','mrp','description'])[['product_name','price','mrp', 'diff_per', 'color', 'brand']]
df = brands_info
df = df[df['diff_per']>0]
category_plots(df['category'].unique(), df, y='category',swarm = True, money = 'diff_per', xtitle='% discount',
               figsize1=(12,5), figsize2=(12,4), figsize3=(10,2))
df[df['diff_per']>60].groupby('category').size()
#**Question 25:** Sales! Up to 50%. What percentage disconts are offered by each brand for each category?
#* Around 20% (709) of panties have a discount >=20% (with 10% (348) of panties having >=50% discount)
#* 15% (190) of bras, 24% (193) of bralettes, 14% (32) of swimsuits and 12% (78) of activewear have a >=50% discount.
#stacked_all(brands_info, 'category', name = 'All Brands', discount_cat='diff_per_categories')
def perform_viz(category, figsize1=(12,4), figsize2=(12,4), figsize3=(10,3)):
    df = brands_info
    df1 = df[df['category']==category]
    stacked_all_brands(df1, discount_cat='diff_per_categories')
    df1 = df[df['diff_per']>0]
    category_plots([category], df1, swarm = True, money = 'diff_per', xtitle = '% discount',
               figsize1=figsize1, figsize2=figsize2, figsize3=figsize3)
    
perform_viz('panties')
perform_viz('bras',figsize1=(12,4), figsize2=(20,3), figsize3=(10,5))
perform_viz('bralettes', figsize1=(12,4), figsize2=(20,3), figsize3=(10,3))
perform_viz('activewear', figsize1=(12,4), figsize2=(20,3), figsize3=(10,3))
perform_viz('pack', figsize1=(12,4), figsize2=(20,3), figsize3=(10,3))
perform_viz('sleepwear', figsize1=(12,4), figsize2=(20,3), figsize3=(10,3))
cats = df['category'].unique()
cats = set(cats)- set(['panties', 'bras', 'activewear', 'bralettes', 'pack', 'sleepwear'])
#df1 = brands_info[brands_info['brand'].isin(brands)]
df1 = df[df['category'].isin(cats)]
stacked_all_brands(df1, discount_cat='diff_per_categories')

df1 = df[df['diff_per']>0]
category_plots(cats, df1, swarm = True, money = 'diff_per', xtitle = '% discount',
              figsize1=(12,4), figsize2=(20,3), figsize3=(10,3))
df = brands_info[brands_info['brand'].isin(['Calvin Klein', "Victoria's Secret"])]
df = df[df['product_category']!='Other']
fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(15,6))
ax = sns.boxplot(y = 'category', x="price", hue='brand', data=df, ax=axes, palette = [light_purple[2], light_purple[5]])
#ax = sns.swarmplot(y = 'category', hue = 'brand', x="mrp", data=df, ax = axes, color = 'grey')
_= ax.tick_params(axis='both', which='major', labelsize=13)
ax.set_xlabel('price', fontsize=13)
ax.set_ylabel('category', fontsize=13)
ax.xaxis.grid(True)
fig.savefig('calvin_victoria_boxplot.png')

fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(15,6))
ax = sns.violinplot(y="price", hue ='brand', x = 'category',cut = 0, split=True, data=df, ax = axes, palette = [light_purple[2], light_purple[5]])
ax.yaxis.grid(True)
_= ax.tick_params(axis='both', which='major', labelsize=13)
ax.set_ylabel('price', fontsize=13)
ax.set_xlabel('category', fontsize=13)
fig.savefig('calvin_victoria_violin.png')

barplots_extra('price', df, (6,10), hue = 'brand', x='category', palette = [light_purple[2], light_purple[5]])
#brands_info.to_csv('data/final_data.csv')