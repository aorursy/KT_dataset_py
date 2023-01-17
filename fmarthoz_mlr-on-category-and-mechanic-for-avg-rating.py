import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import sklearn

import statsmodels.api as sm

import seaborn as sns

from subprocess import check_output

import squarify

print(check_output(["ls", "../input"]).decode("utf8"))



my_df=pd.read_csv('../input/bgg_db_2017_04.csv',encoding = "ISO-8859-1")

#Let's get rid of the games which have a 'none' value in mechanic column

my_df=my_df.drop(my_df[(my_df['category']=='none')].index)

my_df=my_df.drop(my_df[(my_df['mechanic']=='none')].index)

#creating list of unique mechanics and categories

# splitting categories and mechanics into columns

cat={}

mech={}

for label, text in my_df.iterrows():

    liste_cat=text['category'].split(',')

    liste_mec=text['mechanic'].split(',')

    # clean white spaces

    for i,txt in enumerate(liste_cat):

        txt=txt.strip()

        if txt in cat:

            cat[txt]=cat[txt]+1

        else:

            cat[txt]=1

    for j,t in enumerate(liste_mec):

        t=t.strip()

        if t in mech:

            mech[t]=mech[t]+1

        else:

            mech[t]=1





# create all new cat columns and fill them with zeroes

for key in cat:

    my_df[key]=0

for key in mech:

    my_df[key]=0

    

#Now add the '1'    

for label, text in my_df.iterrows():

    liste_cat=text['category'].split(',')

    liste_mec=text['mechanic'].split(',')

    # clean white spaces

    for i,txt in enumerate(liste_cat):

        txt=txt.strip()

        my_df.loc[label,txt]=1

    for j,t in enumerate(liste_mec):

        t=t.strip()

        my_df.loc[label,t]=1
#Running the multiple LM on categories with rating as a response variable

X = my_df.iloc[:,21:] 

y = my_df["avg_rating"]

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

predictions = model.predict(X)

summ=model.summary()

print('Getting all the coefficients greater than |.25|')

coeff_df=model.params[abs(model.params)>0.25]

pval_df=model.pvalues[abs(model.params)>0.25]

results_df=pd.concat([coeff_df,pval_df],axis=1)

results_df.columns=["coeff","p-values"]

print(results_df)

print('R-squared: '+str(model.rsquared))
#are the ratings normally distributed? Doesn't look like it...



plt.subplot(2,1,1)

plt.hist(my_df.loc[:,"avg_rating"],bins=7)

plt.title("histogram of avg rating")

plt.subplot(2,1,2)

plt.hist(np.log(my_df.loc[:,"avg_rating"]),bins=7)

plt.title("histogram of log(avg rating)")

plt.show();



#Checking normality in resp var for linear regression model



fig = sm.qqplot(my_df.loc[:,"avg_rating"])

fig2 = sm.qqplot(np.log(my_df.loc[:,"avg_rating"]))

plt.show()
sns.regplot(x="weight",y="avg_rating",data=my_df, marker="+",color='b',order=2)

plt.show()
top100=my_df.nlargest(100,'avg_rating')

des={}

for label, txt in top100.loc[:,'designer'].iteritems():

#split and remove spaces

    kwds_list=txt.split(", ")

    for a in kwds_list:

        if a in des:

            des[a]=des[a]+1

        else:

            des[a]=1

count=1

for w in sorted(des, key=des.get, reverse=True):

    print(w, des[w])

    if(count==3):

        break

    count=count+1

    
df=pd.DataFrame.from_dict(cat,orient='index')

df.columns=['nber']

squarify.plot(sizes=df['nber'], label=df.index, alpha=.8 )

plt.axis('off')

plt.show()



print("Finding games similar to Pandemic:")



game=0

pand=my_df.iloc[game,20:]



#Recreate a matrix with the scores

df_scores=np.abs(my_df.iloc[:,20:]-pand)

df_scores.insert(0,column='names',value=my_df.loc[:,'names'])

df_scores['sum']=df_scores.iloc[:,3:].sum(axis=1)



print(df_scores.nsmallest(10,columns='sum').loc[:,['names','sum']])



print("Now finding games similar to Terra Mystica:")



game=3

pand=my_df.iloc[game,20:]



#Recreate a matrix with the scores

df_scores=np.abs(my_df.iloc[:,20:]-pand)

df_scores.insert(0,column='names',value=my_df.loc[:,'names'])

df_scores['sum']=df_scores.iloc[:,3:].sum(axis=1)



print(df_scores.nsmallest(10,columns='sum').loc[:,['names','sum']])
