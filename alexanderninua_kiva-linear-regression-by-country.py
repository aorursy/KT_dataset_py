import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
sns.set_palette('pastel')
#The following line of code enables the automatic graph display in Jupyter notebook

%matplotlib inline

#The rest of modules used in the workbook are going to be loaded when they are needed
df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv")
pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', 30)
df.info()  
df.head(3)
ax = sns.heatmap(df.drop(['id', 'partner_id'], axis = 1).corr(), cmap = 'coolwarm', annot = True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.xticks(rotation=45)
# As we can see from the heatmap above, some of the factors, such as funded_amount and lender_count present a strong correlation.
# This correlation can be interpreted in the following way: the bigger the sum of the loan, the more lenders it needs to be funded
df[['funded_amount', 'lender_count']].corr()
#extracting all the unique country names in a single list
countries = list(df['country'].unique())
#Each country is introduced as a key of the dictionnary with a corresponding dataset as a value 
countrydict = {elem : pd.DataFrame() for elem in countries}
for key in countrydict.keys():
    countrydict[key] = df[:][df.country == key]
#Each dataset will have to drop the columns unnecessary to our future work
for key in countrydict:
    countrydict[key].drop(['id','activity', 'use', 'country_code','region','currency','partner_id','posted_time','disbursed_time','funded_time','tags','date','loan_amount','country', 'loan_amount'], axis=  1, inplace = True)
#This function will rework the gender column. As it is a simplification it has to be taken with a grain of salt
def genalloc(x):
    x = str(x)
    #The following line transforms gender strings in lists 'female, female, female, male' -> ['female','female','female','male']
    x = [x.strip() for x in x.split(',')]
    
    #monogender lists keep their value as a string
    if len(x) == 1:
        if x[0] == 'male':
            return 'male'
        elif x[0] == 'female':
            return 'female'
    #longer lists get a new string assigned based on their gender composition
    if len(x) > 1:
        if all(i in x for i in ['male', 'female']):
            return 'mixed'
        elif x[0] == 'male':
            return 'men'
        elif x[0] == 'female':
            return 'women'
#The fuctio is then applied to the 'genders' column
for key in countrydict:
    countrydict[key].dropna(inplace = True)
    countrydict[key]['borrower_genders'] = countrydict[key]['borrower_genders'].apply(lambda x: genalloc(x))
countrydict['Pakistan']['borrower_genders'].unique()
#Creating data dummies for the categorical values of our datasets
for key in countrydict:
    sex = pd.get_dummies(countrydict[key]['borrower_genders'],drop_first= True)
    ints = pd.get_dummies(countrydict[key]['repayment_interval'],drop_first= True)
    sec = pd.get_dummies(countrydict[key]['sector'],drop_first= True)
    countrydict[key].drop(['borrower_genders','repayment_interval','sector'], axis = 1, inplace = True)
    countrydict[key] = pd.concat([countrydict[key], sex, ints,sec],axis = 1)
#Unfortunately, not all datasets have the same column entries. 
#The following dict has countries as keys with their respective dataset shapes as values
dfshapes = {}
for key in countrydict:
    dfshapes[key] = pd.DataFrame(index = countrydict[key].columns.drop('funded_amount')).shape
    
print(max(dfshapes, key=dfshapes.get))
print(dfshapes[max(dfshapes, key=dfshapes.get)])
print('\n')
print(min(dfshapes, key=dfshapes.get))
print(dfshapes[min(dfshapes, key=dfshapes.get)])

#We can see that the gap between the shapes of countries is rather big
#We will use Kenya's columns as standard for all the countries
#The same number of columns
full_columns = countrydict['Kenya'].columns
#If the columns is absent in a dataframe it is added with empty values
for key in countrydict:
    for x in full_columns:
        if x not in countrydict[key]:
            countrydict[key][x] = 0
#The shapes are now normalized
dfshapes = {}
for key in countrydict:
    dfshapes[key] = pd.DataFrame(index = countrydict[key].columns.drop('funded_amount')).shape

print('Kenya \n',dfshapes['Kenya'],'\n','Mauritania\n',dfshapes['Mauritania'])
#We will gather lengths of dataframes in order to weed out country datasets that have a small amount of entries
#The following loops will remove datasets of countries that have less than a thousand entries 
short = []
for k, v in countrydict.items():
    if len(v) < 1000:
        short.append(k)
print(f'There are {len(short)} countries that have fewer than 1000 entries')
for x in short:
    countrydict.pop(x)

#Importing necessary modules from sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
#Initiating dataframes that will later hold the coefficients for every model as well as the different metrics measuring models' efficiency
coefficients = pd.DataFrame(index = full_columns[1:])
errors = pd.DataFrame(index = ['Mean Absolute Error', 'Mean Squared Error', 'Mean Squared Error Root','R2 Score'])
#The following loop iterates through each dataset in countrydict
#Sklearn algorithms split data in test/train sets
#Linear regression model is initiated and fitted for the required information
#Both coefficients and metrics for each country are later appended to the respective dataframes

for key in countrydict:
    X = countrydict[key].drop('funded_amount', axis = 1)
    y = countrydict[key]['funded_amount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    lm = LinearRegression()
    lm.fit(X_train,y_train)
    preds = lm.predict(X_test)
    err_list = []
    err_list.append(metrics.mean_absolute_error(y_test,preds))
    err_list.append(metrics.mean_squared_error(y_test, preds))
    err_list.append(np.sqrt(metrics.mean_squared_error(y_test, preds)))
    err_list.append(metrics.r2_score(y_test, preds))
    coefficients[str(key)] = lm.coef_
    errors[str(key)] = err_list
#Below we can see a dataframe with metrics calculated for each country's regression model
errors = errors.transpose()
errors.nlargest(n = 75 ,columns = 'R2 Score')
#The R2 scores range from around 0.55 to 0.98 with a sigificant concentration around 0.9
sns.distplot(errors['R2 Score'], bins = 25, kde = False)
#This function will add the entry count for each country
#This will enable us to see the relation between the R2 Score and the number of entries the regression is based on
df_ccount = df.groupby('country').count()
def country_count(x):
    return df_ccount.loc[str(x), 'id']
#Creating the Count column which will display the number of entries for each country
errors.reset_index(level = 0, inplace = True)
errors['Count'] = errors['index'].apply(lambda x: country_count(x))
errors.set_index('index', inplace = True)
errors.nlargest(n = 10 ,columns = 'R2 Score')
#As we can see, there is no appearent correlation between the number of entries and the R2 Score
sns.jointplot(x = 'R2 Score', y = 'Count', data = errors, kind = 'hex')
#We will also search for correlation of the R2 core with the standard deviation of the funded amount
#It is possible to hypothesize that, in this case, a good R2 Score correlates with low standard deviation
def df_std(x):
    return df[df['country'] == str(x)]['funded_amount'].std()
errors.reset_index(level = 0, inplace = True)
errors['Standard Deviation'] = errors['index'].apply(lambda x: df_std(x))
errors.set_index('index', inplace = True)
#As we can see on the plot below,there is no apparent correlation between the R2 Score and the Standard Deviation
sns.jointplot(x = 'R2 Score', y = 'Standard Deviation', data = errors)
#Countries with highest and lowest R2 scores
top_R2 = errors.nlargest(n = 75 ,columns = 'R2 Score')['R2 Score'].reset_index(level=0, inplace=False)
R2_compare = pd.concat([top_R2[:5], top_R2[-5:]])
sns.catplot(x = 'index', y = 'R2 Score', data = R2_compare, aspect = 4, kind = 'bar')
plt.title('Countries with highest and lowest R2 Scores',fontsize = 20)
plt.xlabel('Country', fontsize = 15)
plt.ylabel('R2 Score', fontsize = 15)
#The following Coefficients table show coefficients that went into constructing each regression for each country 
coefficients = coefficients.transpose()
coefficients.head(5)
country_sectors = pd.DataFrame(index = coefficients.columns[9:], columns = ['Country Min', 'Min', 'Country Max','Max'])

for x in coefficients.columns[9:]:
    country_sectors.loc[str(x)]['Country Min'] = coefficients[coefficients[str(x)] == coefficients[str(x)].min()].index[0]
    country_sectors.loc[str(x)]['Min'] = coefficients[coefficients[str(x)] == coefficients[str(x)].min()][str(x)][0]
    country_sectors.loc[str(x)]['Country Max'] = coefficients[coefficients[str(x)] == coefficients[str(x)].max()].index[0]
    country_sectors.loc[str(x)]['Max'] = coefficients[coefficients[str(x)] == coefficients[str(x)].max()][str(x)][0]
country_sectors