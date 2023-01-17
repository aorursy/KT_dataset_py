import pandas as pd 

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm
college_df = pd.read_csv('../input/us-news-and-world-reports-college-data/College.csv')
#check for any null values in dataset

college_df.isnull().values.any()
#checking length of dataset

len(college_df)
college_df.info()
#setting the name of the college as the index

college_df = college_df.set_index('Unnamed: 0')
college_df.describe()
#obtaining index of cell having a graduation rate greater than 100

college_df[college_df['Grad.Rate'] >100].index
#setting the value to 100 using loc indexing

college_df.loc[(college_df[college_df['Grad.Rate'] >100].index), 'Grad.Rate'] = 100
#checking the operation went through

college_df[college_df['Grad.Rate'] >100]
dummies = pd.get_dummies(college_df.Private)
merged = pd.concat([college_df,dummies], axis = 1)
final = merged.drop(['Private','No'], axis = 1)
X = final.drop("Grad.Rate",1)   #Feature Matrix

y = final["Grad.Rate"]          #Target Variable



#Adding constant column of ones, mandatory for sm.OLS model, default is added in the front, this is a dummy variable

#statsmodels api doesnt take the bias variable into reference

X_1 = sm.add_constant(X)
#Fitting sm.OLS model

model = sm.OLS(y,X_1).fit()

model.pvalues
#Backward Elimination

cols = list(X.columns)

pmax = 1 #placeholder for new p-value max

while (len(cols)>0):

    p= []

    X_1 = X[cols]

    X_1 = sm.add_constant(X_1)

    model = sm.OLS(y,X_1).fit()

    p = pd.Series(model.pvalues.values[1:],index = cols) #not idexing the constant column     

    pmax = max(p)

    feature_with_p_max = p.idxmax()

    if(pmax>0.05):

        cols.remove(feature_with_p_max)

    else:

        break

selected_features_BE = cols

print(selected_features_BE)
index_selected_features_BE = []

for col in X[selected_features_BE]:

    index_selected_features_BE.append(X.columns.get_loc(col))
final[selected_features_BE]
x = final.loc[:, final.columns != 'Grad.Rate'].values

y = final.loc[:, 'Grad.Rate'].values
X_train, X_test, Y_train, Y_test = train_test_split(x[:,index_selected_features_BE], y, test_size = 0.8, random_state = 0)
#normalization of data 

scaler = StandardScaler()

X_train[:,:8] = scaler.fit_transform(X_train[:,:8])

X_test[:,:8] = scaler.transform(X_test[:,:8])
model = LinearRegression()

model.fit(X_train,Y_train)

print('Model score: '+str(model.score(X_test,Y_test)))