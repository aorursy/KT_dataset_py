#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model, metrics


import statsmodels.api as sm
import seaborn as sns
%matplotlib inline
import warnings; warnings.simplefilter('ignore')
from subprocess import check_output
# Any results you write to the current directory are saved as output.
filename = check_output(["ls", "../input"]).decode("utf8").strip()
df_raw = pd.read_csv("../input/" + filename, thousands=",")
df_raw.head()
#check correlation
df_corr = df_raw[['grams', 'quality', 'btc_price', 'cost_per_gram', 'cost_per_gram_pure', 'rating']].corr()

print (df_corr)

sns.heatmap(df_corr, cmap="Blues")
#visual on categorical variable ship from
sns.factorplot(
    x='ships_from',
    y='cost_per_gram',
    data=df_raw, 
    kind='box', 
    size = 3,
     aspect=5

)
fg = sns.FacetGrid(df_raw[['quality', 'cost_per_gram', 'ships_from']], hue='ships_from',size=5, aspect=3)
fg.map(plt.scatter, 'quality', 'cost_per_gram').add_legend()

#Plot data distributions

# Set up the matplotlib figure
f, axes = plt.subplots(2, 3, figsize=(10, 10), sharex=False, sharey=False)

sns.distplot(df_raw['cost_per_gram_pure'], ax=axes[0, 0])
sns.distplot(df_raw['cost_per_gram'], ax=axes[0, 1])
sns.distplot(df_raw['quality'], ax=axes[0, 2])
sns.distplot(df_raw['btc_price'], ax=axes[1,0]) 
sns.distplot(df_raw['grams'], ax=axes[1,1]) 
sns.distplot(df_raw['rating'], ax=axes[1,2]) 


df_raw[['grams', 'quality', 'btc_price', 'cost_per_gram', 'cost_per_gram_pure', 'rating']].skew()  # Check skewness
df_raw[['grams', 'quality', 'btc_price', 'cost_per_gram', 'cost_per_gram_pure', 'rating']].kurt()  # Check kurtosis
#Attemp to fix kurtosis and have better normal distribution by appling log function

df_raw['btc_price_log'] = df_raw['btc_price'].apply(np.log)

f, axes = plt.subplots(1, 2, figsize=(10, 7), sharex=False, sharey=False)

sns.distplot(df_raw['btc_price'], ax=axes[0])
sns.distplot(df_raw['btc_price_log'], ax=axes[1])
df_raw[['btc_price', 'btc_price_log']].skew() # Check kurtosis

print ("grams  unique categories")
print (df_raw['grams'].unique())
#Add dummy categories for grams
df_raw['grams'].unique()
dummy_rank = pd.get_dummies(df_raw['grams'], prefix='grams', prefix_sep='_')
df_combo = df_raw.join(dummy_rank)
print ("quality unique categories")
print (df_raw['quality'].unique())
#Add dummy categories for quality
df_raw['quality'].unique()
dummy_rank = pd.get_dummies(df_raw['quality'], prefix='quality', prefix_sep='_')
df_combo = df_combo.join(dummy_rank)
df_combo.info()
#convert boolean variables (with String elements in it) to numeric to fit the model

ships_to = df_combo.columns
ships_to_t = []
for q in ships_to:
    if 'ships_to_' in q:
        ships_to_t.append(q)
     #   print 1
    elif 'ships_from_' in q:
        ships_to_t.append(q)
ships_to_t.remove('ships_from_to')

for d in ships_to_t:
    df_combo[d] = df_combo[d].apply(lambda x: int(bool(x)))
    


#add intercept
df_combo['intercept'] = 1


#collect all names of columns' features in a list
X_columns = df_combo.columns
X_btc_price_combo_col = []
for q in X_columns:
    if 'grams_' in q:
        X_btc_price_combo_col.append(q)
      #  print q

for w in X_columns:
    if 'quality_' in w:
        X_btc_price_combo_col.append(w)
 #       print q

for q in X_columns:
    if 'ships_from_' in q:
        X_btc_price_combo_col.append(q)
    elif 'ships_to_' in q:
        X_btc_price_combo_col.append(q)
        
X_btc_price_combo_col.remove('ships_from_to')
X_btc_price_combo_col.append('escrow')    
X_btc_price_combo_col.append('intercept')
print (X_btc_price_combo_col)
X_btc_price_combo = df_combo[X_btc_price_combo_col]
y_btc_price_combo = df_combo["btc_price_log"]


#no need in calssical as we use lasso to suppress all potentially multicollinearity features
#model = sm.OLS(y_btc_price_combo, X_btc_price_combo)
#results = model.fit()
#print(results.summary())
#results.resid

#Use GS to identify the best alpha parameter 
from sklearn import grid_search


alphas = np.logspace(-10, 10, 21)
gs = grid_search.GridSearchCV(
    estimator= linear_model.Lasso(),
    param_grid={'alpha': alphas},
    scoring='mean_squared_error')

gs.fit(X_btc_price_combo, y_btc_price_combo)

print (gs.best_score_ )

print (gs.best_estimator_ )
#Apply the identified alpha and use the model outcome in features analysis 
lm = linear_model.Lasso(alpha=0.0001,positive=True).fit(X_btc_price_combo, y_btc_price_combo)
#print (lm.coef_)
#print (lm.intercept_)
print ("~~~ Lasso ~~~")
print ('Lasso MSE: ', metrics.mean_squared_error(y_btc_price_combo, lm.predict(X_btc_price_combo)))
print ('Lasso R2:', lm.score(X_btc_price_combo, y_btc_price_combo))

# some quality metrics. Plot true values versus residual
#rss = sum((lm.predict(X_btc_price_combo)-df_combo["btc_price_log"])**2)
#print rss
plt.scatter(lm.predict(X_btc_price_combo), df_combo["btc_price_log"] - lm.predict(X_btc_price_combo), color="g")
plt.xlabel("True Values")
plt.ylabel("Predicted Values resid")
#plt.show()
## rate of BTC to AUD in July 2017
CurrencyRate = 3050
#create a reporting table based on prediction of 1 gram of quality 75% from different countries
Country = []
Cost = []  
df_predict = pd.DataFrame({'Predict' : 0
                           }, index = X_btc_price_combo_col).T

#df_predict['escrow'] = 1
df_predict['quality_75.0'] = 1
df_predict['grams_1.0'] = 1
df_predict['ships_to_GB'] = 1


for q in X_btc_price_combo_col:
    #go through the loop of ships_from_ prefix predictors 
    if 'ships_from_' in q:
        df_predict[q] = 1
        Country.append(q)
        Cost.append(np.exp(lm.predict(df_predict)) * CurrencyRate)
        df_predict[q] = 0
       # print q, np.exp(lm.predict(df_predict)) * CurrencyRate

df_plt_by_country = pd.DataFrame({ 'Country' : Country,
                       'Cost' : Cost
                         })
df_plt_by_country['Cost_num'] = df_plt_by_country['Cost'].astype(float)


#Report predicted Cost by Country
sns.factorplot(
    y='Country',
    x='Cost_num',
    data=df_plt_by_country.sort_values('Cost_num',ascending=False ), 
    kind='bar', 
    size = 5,
     aspect=1,
orient = 'h',
    color = 'b'
)
Grams = []
Cost = []  


df_predict = pd.DataFrame({'Predict' : 0
                           }, index = X_btc_price_combo_col).T

CurrencyRate = 3050
df_check = pd.DataFrame({ 'btc' : float,
                       'Cost' : float
                         }, index = [1])
df_check['Cost'] = np.exp(lm.predict(df_predict)) * CurrencyRate
df_check['btc'] = np.exp(lm.predict(df_predict))
df_check['log'] = lm.predict(df_predict)
df_check['rate'] = CurrencyRate

for q in X_btc_price_combo_col:
    #go through the loop of ships_from_ prefix predictors 
    if 'grams_' in q:
        df_predict[q] = 1
        Grams.append(q)
        Cost.append(np.exp(lm.predict(df_predict)) * CurrencyRate)
        df_predict[q] = 0
       # print q, np.exp(lm.predict(df_predict)) * CurrencyRate

df_plt_by_grams = pd.DataFrame({ 'Grams' : Grams,
                       'Cost' : Cost
                         })
#Convert to numeric and calcualte cost per gram
df_plt_by_grams['Cost_num'] = df_plt_by_grams['Cost'].astype(float)
df_plt_by_grams[['Cost_num', 'Grams']]
df_plt_by_grams['Grams_num'] = df_plt_by_grams['Grams'].map(lambda x: str(x)[6:]).astype(float)
df_plt_by_grams['Cost_per_gram'] =df_plt_by_grams['Cost_num'] /  df_plt_by_grams['Grams_num'] 

#print df_plt_by_grams
sns.factorplot(
    y='Grams',
    x='Cost_per_gram',
    data=df_plt_by_grams.sort_values('Grams_num',ascending=True ), 
    kind='bar', 
    size = 15,
     aspect=1,
orient = 'h'
    ,
    color = 'b'
)
#create a reporting table based on prediction of 1 gram  by quality variables in Australia
Quality = []
Cost = []  
df_predict = pd.DataFrame({'Predict' : 0
                           }, index = X_btc_price_combo_col).T
#df_predict['escrow'] = 1
#df_predict['ships_to_AU'] = 1
#df_predict['ships_from_AU'] = 1
#df_predict['grams_1.0'] = 1

for q in X_btc_price_combo_col:
    #go through the loop of quality_ prefix predictors 
    if 'quality_' in q:
        df_predict[q] = 1
        Quality.append(q)
        Cost.append((lm.predict(df_predict)))
        df_predict[q] = 0
       # print q, np.exp(lm.predict(df_predict)) * CurrencyRate

    
df_plt_by_qa = pd.DataFrame({ 'quality' : Quality,
                       'Cost' : Cost
                         })

df_plt_by_qa['Cost_num'] = df_plt_by_qa['Cost'].astype(float)


sns.factorplot(
    y='quality',
    x='Cost_num',
    data=df_plt_by_qa.sort_values('quality',ascending=False ), 
    kind='bar', 
    size = 5,
     aspect=1,
orient = 'h'
    ,
    color = 'b'
)
df_coef = pd.DataFrame({'Coef' : lm.coef_,
                         'Inercept' : lm.intercept_,
                        'Name': X_btc_price_combo_col
                           }, index = X_btc_price_combo_col)

sns.factorplot(
    y='Name',
    x='Coef',
    data=df_coef, 
    kind='bar', 
    size = 40,
     aspect=1,
#orient = 'v',
     legend_out=True,
    color = 'b'
 #   palette = 'RdBu'
)
