import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
wine = pd.read_csv("../input/winemag-data-130k-v2.csv")

print(wine.info())
wine = wine[['country', 'points', 'price', 'taster_name', 'variety', 'winery', 'description']]

wine = wine.dropna().reset_index(drop=True)

print(wine.describe())
sns.jointplot(x='price', y='points', data=wine, facecolors='none', edgecolors='darkblue', alpha=0.1)

plt.show()
# We take the logarithm of the price and points columns



log_score = np.log(wine[['price', 'points']])

log_score = log_score.dropna().reset_index(drop=True)

log_score.columns = ['log_price', 'log_score']



# We visualize our DataFrame

log_score.head()
# Scatter plot



sns.jointplot(x='log_price', y='log_score', data=log_score, facecolors='none', edgecolors='darkblue', alpha=0.05)

plt.show()
# Linear regression: 95% confidence interval (ci)



sns.regplot(x='log_price', y='log_score', data=log_score, scatter_kws={'alpha':0.05}, ci=95)

plt.show()
# Correlation coefficient: 



corr = np.corrcoef(wine['points'], wine['price'])

corr2 = np.corrcoef(log_score['log_score'], log_score['log_price'])

print('Data correlation coefficient: %.4f \nLog-data correlation coefficient: %.4f' % (corr[0][1], corr2[0][1]))
# Price quartiles



quart1 = wine[wine.price < wine.price.quantile(.25)].reset_index(drop=True)

quart1 = quart1.dropna().reset_index(drop=True)



quart2 = wine[(wine.price < wine.price.quantile(.50)) & (wine.price >= wine.price.quantile(.25))].reset_index(drop=True)

quart2 = quart2.dropna().reset_index(drop=True)



quart3 = wine[(wine.price < wine.price.quantile(.75)) & (wine.price >= wine.price.quantile(.50))].reset_index(drop=True)

quart3 = quart3.dropna().reset_index(drop=True)



quart4 = wine[wine.price >= wine.price.quantile(.75)].reset_index(drop=True)

quart4 = quart4.dropna().reset_index(drop=True)
# Figure parameters



plt.figure(figsize=(20,10))



#------

plt.subplot(2, 2, 1)



plt.title('Quartile 1', fontsize=20)

sns.distplot( quart1['points'], color='green', kde=False)

plt.axvline(np.mean(wine.points), 0,1, linestyle='--', color='black', label='Mean score')

plt.axvline(np.mean(quart1.points), 0,1, linestyle='--', color='green', label='Q1 mean score')

plt.legend(fontsize=15)



#------

plt.subplot(2, 2, 2)

plt.title('Quartile 2', fontsize=20)

sns.distplot( quart2['points'], color='gold', kde=False)

plt.axvline(np.mean(wine.points), 0,1, linestyle='--', color='black', label='Mean score')

plt.axvline(np.mean(quart2.points), 0,1, linestyle='--', color='gold', label='Q2 mean score')

plt.legend(fontsize=15)



#------

plt.subplot(2, 2, 3)

plt.title('Quartile 3', fontsize=20)

sns.distplot( quart3['points'], color='red', kde=False)

plt.axvline(np.mean(wine.points), 0,1, linestyle='--', color='black', label='Mean score')

plt.axvline(np.mean(quart3.points), 0,1, linestyle='--', color='red', label='Q3 mean score')

plt.legend(fontsize=15)



#------

plt.subplot(2, 2, 4)

plt.title('Quartile 4', fontsize=20)

sns.distplot( quart4['points'], color='skyblue', kde=False)

plt.axvline(np.mean(wine.points), 0,1, linestyle='--', color='black', label='Mean score')

plt.axvline(np.mean(quart4.points), 0,1, linestyle='--', color='skyblue', label='Q4 mean score')

plt.legend(fontsize=15)



plt.show()
# We now compare all quartiles on the same graph



# Figure parameters

plt.figure(figsize=(20,10))



plt.title('Score distribution per price quantile', fontsize=30)

sns.kdeplot( quart1['points'], color='green',   label='Quartile 1', shade=True)

sns.kdeplot( quart2['points'], color='gold',    label='Quartile 2', shade=True)

sns.kdeplot( quart3['points'], color='red',     label='Quartile 3', shade=True)

sns.kdeplot( quart4['points'], color='skyblue', label='Quartile 4', shade=True)

plt.axvline(np.mean(wine.points), 0,1, linestyle=':', color='black', label='Mean score')

plt.legend(fontsize=15)



plt.show()
# This function will be used to obtain the list of different critics who rated the wine



def GetUniqueParameterValues(parameter_of_interest):

    '''

    Parameters:

    ----------------------------------------

    

    parameter_of_interest: 

        String - Which parameter we wish to find. e.g. 'country', 'variety'

    

    '''

    param_data = wine[[parameter_of_interest, 'points']]

    param_data = param_data.dropna().reset_index(drop=True)

    

    param_values = param_data[parameter_of_interest]

    param_values = pd.DataFrame(param_values).drop_duplicates(keep='first').reset_index(drop=True)

    param_values = np.array(param_values)

    

    param_list = []

    for i in range(len(param_values)):

        param_list.append(param_values[i][0])

    return param_list
# This function gets the data asociated to a particular parameter

# This function will be used within the CalculateCorrelation function below



def GetParameterData(parameter_of_interest, param_value):

    '''

    Parameters:

    ----------------------------------------

    

    parameter_of_interest: 

        String - Which parameter we wish to find. e.g. 'country', 'variety'

    

    param_value:

        String - Specific value of the parameter e.g. 'Italy', 'Cabernet Sauvignon', etc.

    

    '''

    

    res = wine.loc[wine[parameter_of_interest] == param_value]

    res = res[['price', 'points']]

    res = res.dropna().reset_index(drop=True)

    return res
# This function calculates the correlation coefficient between price and score

# of our wines according to their country, winery and variety



def CalculateCorrelation(parameter_of_interest, param_value_list):

    '''

    Parameters:

    ----------------------------------------

    

    parameter_of_interest: 

        String - Which parameter we wish to find. e.g. 'country', 'variety'

    

    param_value_list:

        List - Lists of string elements of the parameters we are analyzing. e.g. list of wineries, 

               list of countries

    

    '''

    res = []

    for i in range(len(param_value_list)):

        data_ = GetParameterData(parameter_of_interest, param_value_list[i])

        corr_coef = np.corrcoef(data_['price'], data_['points'])[0][1]

        temp = param_value_list[i], corr_coef

        res.append(temp)

    res = pd.DataFrame(res)

    res.columns = [parameter_of_interest, 'corr_coef']

    res = res.sort_values(by=['corr_coef'], ascending=True).reset_index(drop=True) #DEBERIA SER ascending=False

    return res
# We visualize the top 19 wineries, countries and varieties in terms of numer of data entries



wrs = wine['winery'].value_counts()

ctr = wine['country'].value_counts()

vrt = wine['variety'].value_counts()

print(wrs.head(19), '\n------------------------------\n', ctr.head(19) 

                  , '\n------------------------------\n', vrt.head(19))
# We look at the following variables: taster, country, winery, variety



# Tasters which we will be comparing

tasters   = GetUniqueParameterValues('taster_name')



# Countries which we will be comparing

country   = ['US', 'France', 'Italy', 'Spain', 'Portugal', 'Chile', 'Argentina', 'Austria', 'Germany', 'Australia', 

            'New Zealand', 'South Africa', 'Israel', 'Greece', 'Canada', 'Hungary', 'Bulgaria', 'Romania', 

            'Uruguay']



# Wineries which we will be comparing

wineries  = ['Wines & Winemakers', 'DFJ Vinhos', 'Chateau Ste. Michelle', 'Concha y Toro', 

            'Louis Latour', 'Columbia Crest', 'Georges Duboeuf', 'Montes', 'Trapiche', 'Testarossa', 'Santa Ema', 

            'Undurraga', 'Maryhill', 'Chehalem', 'Jean-Luc and Paul Aegerter', 'Chanson Père et Fils', 

            'Seven Hills', "D'Arenberg", 'Georges Vigouroux']



# Varieties which we will be comparing

varieties = ['Pinot Noir', 'Chardonnay', 'Red Blend', 'Cabernet Sauvignon', 'Bordeaux-style Red Blend', 

            'Riesling', 'Sauvignon Blanc', 'Syrah', 'Rosé', 'Malbec', 'Portuguese Red', 'Merlot', 'Sangiovese', 

            'Nebbiolo', 'Tempranillo', 'White Blend', 'Sparkling Blend', 'Zinfandel', 'Pinot Gris']
# Correlation coefficient between price and score:



# a) According to different tasters

tst_corr = CalculateCorrelation('taster_name', tasters)



# b) According to different countries

ctr_corr = CalculateCorrelation('country', country)



# c) According to different wineries

wnr_corr = CalculateCorrelation('winery', wineries)



# d) According to different varieties

vrt_corr = CalculateCorrelation('variety', varieties)
# Visualization of the correlation coefficients



tst_range = range(1,len(tst_corr.index)+1)

my_range = range(1,len(ctr_corr.index)+1)



sns.set()



# Figure parameters

plt.figure(figsize=(15,10))



#------

plt.subplot(2, 2, 1)



plt.hlines(y=my_range, xmin=0, xmax=ctr_corr['corr_coef'], color='skyblue')

plt.plot(ctr_corr['corr_coef'], my_range, "o")



plt.yticks(my_range, ctr_corr['country'])

plt.xlabel('Correlation coefficient')

plt.ylabel('Country')



plt.axvline(1, 0,1, linestyle='--', color='black')

plt.axvline(np.mean(ctr_corr.corr_coef), 0,1, linestyle='--', color='gold')



#------

plt.subplot(2, 2, 2)



plt.hlines(y=my_range, xmin=0, xmax=wnr_corr['corr_coef'], color='skyblue')

plt.plot(wnr_corr['corr_coef'], my_range, "o")



plt.yticks(my_range, wnr_corr['winery'])

plt.xlabel('Correlation coefficient')

plt.ylabel('Winery')



plt.axvline(1, 0,1, linestyle='--', color='black')

plt.axvline(np.mean(wnr_corr.corr_coef), 0,1, linestyle='--', color='gold')



#------

plt.subplot(2, 2, 3)



plt.hlines(y=my_range, xmin=0, xmax=vrt_corr['corr_coef'], color='skyblue')

plt.plot(vrt_corr['corr_coef'], my_range, "o")



plt.yticks(my_range, vrt_corr['variety'])

plt.xlabel('Correlation coefficient')

plt.ylabel('Variety')



plt.axvline(1, 0,1, linestyle='--', color='black')

plt.axvline(np.mean(vrt_corr.corr_coef), 0,1, linestyle='--', color='gold')



#------

plt.subplot(2, 2, 4)



plt.hlines(y=tst_range, xmin=0, xmax=tst_corr['corr_coef'], color='skyblue')

plt.plot(tst_corr['corr_coef'], tst_range, "o")



plt.yticks(tst_range, tst_corr['taster_name'])

plt.xlabel('Correlation coefficient')

plt.ylabel('Taster name')



plt.axvline(1, 0,1, linestyle='--', color='black')

plt.axvline(np.mean(tst_corr.corr_coef), 0,1, linestyle='--', color='gold')



plt.tight_layout()

plt.show()
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# Score quartiles



score_quart4 = wine[wine.points >= wine.points.quantile(.75)].reset_index(drop=True)

score_quart4 = score_quart4.dropna().reset_index(drop=True)



score_quart3 = wine[(wine.points < wine.points.quantile(.75)) & (wine.points >= wine.points.quantile(.50))].reset_index(drop=True)

score_quart3 = score_quart3.dropna().reset_index(drop=True)



score_quart2 = wine[(wine.points < wine.points.quantile(.50)) & (wine.points >= wine.points.quantile(.25))].reset_index(drop=True)

score_quart2 = score_quart2.dropna().reset_index(drop=True)



score_quart1 = wine[wine.points < wine.points.quantile(.25)].reset_index(drop=True)

score_quart1 = score_quart1.dropna().reset_index(drop=True)



# Price quantiles were defined previously
# Wine descriptions by PRICE



qt1_txt = str(quart1.description)

qt2_txt = str(quart2.description)

qt3_txt = str(quart3.description)

qt4_txt = str(quart4.description)



stopwords = set(STOPWORDS)



# We remove words that are common among all descriptions, in order to see if there is one characteristic

# that differentiates these wines

stopwords.update(["drink", "now", "wine", "flavor", "flavors", "aroma", "aromas", "blend", "note", "notes"])



# Figures



plt.figure(figsize=(15,15))



plt.subplot(2, 2, 1)

plt.title('Q1 by price (cheapest)', fontsize=15)

wordcloud = WordCloud(stopwords=stopwords, background_color='white').generate(qt1_txt)



plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")



#------



plt.subplot(2, 2, 2)

plt.title('Q2 by price', fontsize=15)

wordcloud = WordCloud(stopwords=stopwords, background_color='white').generate(qt2_txt)



plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")



#------



plt.subplot(2, 2, 3)

plt.title('Q3 by price', fontsize=15)

wordcloud = WordCloud(stopwords=stopwords, background_color='white').generate(qt3_txt)



plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")



#------



plt.subplot(2, 2, 4)

plt.title('Q4 by price (priciest)', fontsize=15)

wordcloud = WordCloud(stopwords=stopwords, background_color='white').generate(qt4_txt)



plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")



plt.tight_layout()

plt.show()
# Wine descriptions by SCORE



qt1_txt = str(score_quart1.description)

qt2_txt = str(score_quart2.description)

qt3_txt = str(score_quart3.description)

qt4_txt = str(score_quart4.description)



stopwords = set(STOPWORDS)



# We remove words that are common among all descriptions, in order to see if there is one characteristic

# that differentiates these wines

stopwords.update(["drink", "now", "wine", "flavor", "flavors", "aroma", "aromas"])



# Figures



plt.figure(figsize=(15,15))



plt.subplot(2, 2, 1)

plt.title('Q1 by score (lowest-scoring)', fontsize=15)

wordcloud = WordCloud(stopwords=stopwords, background_color='white').generate(qt1_txt)



plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")



#------



plt.subplot(2, 2, 2)

plt.title('Q2 by score', fontsize=15)

wordcloud = WordCloud(stopwords=stopwords, background_color='white').generate(qt2_txt)



plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")



#------



plt.subplot(2, 2, 3)

plt.title('Q3 by score', fontsize=15)

wordcloud = WordCloud(stopwords=stopwords, background_color='white').generate(qt3_txt)



plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")



#------



plt.subplot(2, 2, 4)

plt.title('Q4 by score (highest-scoring)', fontsize=15)

wordcloud = WordCloud(stopwords=stopwords, background_color='white').generate(qt4_txt)



plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")



plt.tight_layout()

plt.show()