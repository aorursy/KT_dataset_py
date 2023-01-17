# numpy and pandas for data manipulation

import pandas as pd

import numpy as np



# sklearn preprocessing for dealing with categorical variables

from sklearn.preprocessing import LabelEncoder



# File system management

import os



# Suppress warnings

import warnings

warnings.filterwarnings('ignore')



# matplotlib and seaborn for plotting

import matplotlib.pyplot as plt

import seaborn as sns
# List of files available.

print(os.listdir('../input/home-credit-default-risk/'))
# Training data

app_train = pd.read_csv('../input/home-credit-default-risk/application_train.csv')

print('Training data shape : ', app_train.shape)

app_train.head()
# Testing data features

app_test = pd.read_csv('../input/home-credit-default-risk/application_test.csv')

print('Testing data shape : ', app_test.shape)

app_test.head()
app_train['TARGET'].value_counts()
app_train['TARGET'].astype(int).plot.hist()
# Total missing values

mis_val = app_train.isnull().sum()



# Percentage of missing values

mis_val_percent = 100 * mis_val / len(app_train)
len(app_train)
mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
mis_val_table
# Rename the columns

mis_val_table_ren_columns = mis_val_table.rename(

    columns = {

        0:'Missing Values',

        1:'% of Total Values'

    }

)
mis_val_table_ren_columns
# Sort the table by percentage of missing descending

mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

'% of Total Values', ascending=False).round(1)



# Print some summary information

print('Your selected dataframe has ' + str(app_train.shape[1]) + ' columns.\n''There are '+str(mis_val_table_ren_columns.shape[0]) + ' columns that have missing values'

     )
mis_val_table_ren_columns
def missing_values_table(df):

    """

    param : 

        df : pandas DataFrame data, processing target dataframe.

    Return :

        mis_val_table_ren_columns : pandas DataFrame , 

        column data extracted from df that has null data

    """

    # Total missing values

    mis_val = df.isnull().sum()

    

    # Percentage of missing values

    mis_val_percent = 100 * mis_val / len(df)

    

    # Make a table with the results / concat은 지정 데이터를 합쳐준다.

    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    

    # Rename the columns

    mis_val_table_ren_columns = mis_val_table.rename(columns={

        0:'Missing Values',

        1:'% of Total Values'

    })

    

    # Sort the table by percentage of missing descending

    mis_val_table_ren_columns=mis_val_table_ren_columns[

        mis_val_table_ren_columns.iloc[:,1]!=0].sort_values(

        '% of Total Values', ascending=False).round(1)

    

    # Print some summary information

    print('You selected dataframe has '+str(df.shape[1])+' columns.\n'

    'There are '+str(mis_val_table_ren_columns.shape[0])+' columns that have missing values.')

    

    # Return the dataframe with missing information

    return mis_val_table_ren_columns
# Missing values statistics

missing_values = missing_values_table(app_train)

missing_values.head(20)
# Number of each type of column

app_train.dtypes.value_counts()
# Number of unique classes in each object column

app_train.select_dtypes('object').apply(pd.Series.nunique, axis=0)

# pd.Series.nunique 메소드는 각 카테고리 컬럼의 데이터를 분류하는 분류자 수를 반환함 // 아래 참조
# 실제 이렇게 데이터 타입을 선택해서 출력해보면, 각 컬럼별로 어떤 종류의 분류자로 분류 되는지 볼 수 있고, 

# nunique는 기본적으로 null데이터는 제외하고 분류자 갯수를 반환해준다.

app_train.select_dtypes('object')
# Create a label encoder object.

le = LabelEncoder()

le_count = 0
# Iterate through the columns

for col in app_train:

    if app_train[col].dtype == 'object':

        # If 2 or fewer unique categories

        if len(list(app_train[col].unique())) <= 2:

            

            # Train on the training data

            le.fit(app_train[col])

            # Transform both Training and Testing data

            app_train[col] = le.transform(app_train[col])

            app_test[col] = le.transform(app_test[col])

            

            print('%s is processed !' % col)

            # Keep track of how many columns label encoded

            le_count +=1            

            

print('%d columns were label encoded' % le_count)
app_train[['FLAG_OWN_REALTY','FLAG_OWN_CAR','NAME_CONTRACT_TYPE']].head(10)
# one-hot encoding of categorical variables

app_train = pd.get_dummies(app_train)

app_test = pd.get_dummies(app_test)



print('Training Feature shape', app_train.shape)

print('Testing Feature shape', app_test.shape)
train_labels=app_train['TARGET']



# Align the training and testing data, keep only columns present in both dataframes

app_train, app_test = app_train.align(app_test, join='inner', axis=1)



# Add the target back in

app_train['TARGET'] = train_labels
print('Training Feature shape', app_train.shape)

print('Testing Feature shape', app_test.shape)
# original DAYS_BIRTH 데이터 값

app_train[['DAYS_BIRTH']].head(10)
(app_train['DAYS_BIRTH'] / (-365)).describe()
(app_train['DAYS_EMPLOYED']).describe()
(app_train['DAYS_EMPLOYED']).plot.hist(title='Days Employment Histogram');

plt.xlabel('Days Employment')
# DAYS_EMPLOYED 값이 최대값을 갖는 고객 데이터를 모두 가져와서 anom에 저장

anom = app_train[app_train['DAYS_EMPLOYED']==365243]

# DAYS_EMPLOYED 값이 최대값이 아닌 고객 데이터를 모두 가져옴!

non_anom = app_train[app_train['DAYS_EMPLOYED']!=365243]
print('The non-anomalies default on %0.2f%% of loans' % (100 * non_anom['TARGET'].mean()))

print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean()))

print('There are %d anamalous days of employment' % len(anom))
# Create anomalous flag column

# 해당 데이터가 있는 곳엔 True를 반환해줌

app_train['DAYS_EMPLOYED_ANOM'] = app_train['DAYS_EMPLOYED']==365243



# Replace the anomalous values with nan

# numpy의 nan을 사용, 사전형 형태로 데이터가 365243인 곳에 np.nan을 반환해줌!

app_train['DAYS_EMPLOYED'].replace({365243:np.nan}, inplace=True)



# 그럼 다시, 히스토그램을 그려서, 데이터의 가공 상태를 확인

app_train['DAYS_EMPLOYED'].plot.hist(title='Days Employment Histogram after processing ANOM data');

plt.xlabel('Days Employment');



(app_train[['DAYS_EMPLOYED']] / (-365)).describe()
app_test['DAYS_EMPLOYED'].describe()
app_test['DAYS_EMPLOYED_ANOM'] = app_test['DAYS_EMPLOYED'] == 365243

app_test['DAYS_EMPLOYED'].replace({365243:np.nan}, inplace=True)
print('There are %d anomalies in the test data out of %d entries' % 

      (app_test['DAYS_EMPLOYED_ANOM'].sum(), len(app_test)))
app_test[app_test['DAYS_EMPLOYED']==365243]
# 먼저 전체적인 코릴레이션 계수를 이렇게 출력할 수 있다. 디폴트는 피어슨 방법으로 코릴레이션 함

corrmat = app_train.corr()

corrmat
f, ax = plt.subplots(figsize=(20,15))

sns.heatmap(corrmat, vmax=1.0, square=True)
# Find correlations with the target and sort

corrmat = corrmat['TARGET'].sort_values()

# Display correlations

print('Most Positive Correlations : \n', corrmat.tail(15))

print('\n')

print('Most Negative Correlations : \n', corrmat.head(15))
# Find the correlation of the positive days since birth and target

app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])

app_train['DAYS_BIRTH'].corr(app_train['TARGET'])
app_train['DAYS_BIRTH'].describe()
# Set the style of plots

plt.style.use('fivethirtyeight')



# Plot the distribution of ages in years

plt.hist(app_train['DAYS_BIRTH'] / 365, edgecolor = 'k', bins=25) # bins 옵션은 몇개의 막대로 표현할지 정해줌

plt.title('Age of Client')

plt.xlabel('Age (years)')

plt.ylabel('Count')           

           
plt.figure(figsize=(10,8))



# KDE plot of loans that were repaid on time

sns.kdeplot(app_train.loc[app_train['TARGET']==0, 'DAYS_BIRTH'] / 365, label='target == 0')



# KDE plot of loans which were not repaid on time

sns.kdeplot(app_train.loc[app_train['TARGET']==1, 'DAYS_BIRTH'] / 365, label='target == 1')



# Labeling of plot

plt.xlabel('Age (years)')

plt.ylabel('Density')

plt.title('Distribution of Ages')
# Age information into a separate dataframe

age_data = app_train[['TARGET', 'DAYS_BIRTH']]

age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365



# Bin the age data

age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins=np.linspace(20,70, num=11))

age_data.head(10)
np.linspace(20,70, num=11) # 처음 수, 끝 수, 총 숫자 수=11
# Group by the bin and calculate averages

age_groups = age_data.groupby('YEARS_BINNED').mean()

age_groups
age_groups.index.astype(str)
plt.figure(figsize=(8,8))



# Graph the age bins and the average of the target as a bar plot

plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'] )



# Plot labeling

plt.xticks(rotation = 75)

plt.xlabel('Age Group (years)')

plt.ylabel('Failure to Repay (%)')

plt.title('Failure to Repay by Age Group');
# Extract the EXT_SOURCE variables and show correlations

ext_data = app_train[['TARGET', 'EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH']]

ext_data_corrs = ext_data.corr()

ext_data_corrs
plt.figure(figsize=(8,6))



# Heatmap of correlations

sns.heatmap(ext_data_corrs, cmap=plt.cm.RdYlBu_r, vmin=-0.25, annot=True, vmax=0.6)

plt.title('Correlation Heatmap');
plt.figure(figsize=(10,12))



# iterate through the sources

for i, source in enumerate(['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']):

    

    # create a new subplot for each source

    plt.subplot(3,1,i+1) # 3 행 1열 모양으로 잡고, 각 순서대로 플랏 함

    # plot repaid loans

    sns.kdeplot(app_train.loc[app_train['TARGET']==0,source], label='target == 0')

    # plot loans that were not repaid

    sns.kdeplot(app_train.loc[app_train['TARGET']==1,source], label='target == 1')

    

    plt.title('Distribution of %s by Target Value' % source)

    plt.xlabel('%s' % source)

    plt.ylabel('Density');



plt.tight_layout(h_pad = 2.5)

    
# Copy the ext_data for plotting

ext_data.head()
plot_data = ext_data.drop(columns=['DAYS_BIRTH']).copy()
# 데이터 컬럼 확인!

plot_data.head()
# Add in the age of the client in years

plot_data['YEARS_BIRTH'] = age_data['YEARS_BIRTH']

# Drop na values and limit to first 100000 rows

plot_data = plot_data.dropna().loc[:100000, :]
plot_data.shape

# 전체 데이터 수 가 3분의 1로 줄었음.
# Function to calculate correlation coeffiecient between two columns

def corr_func(x,y, **kwargs):

    r = np.corrcoef(x,y)[0][1]

    ax = plt.gca()

    ax.annotate('r = {:.2f}'.format(r),

                xy=(.2, .8), xycoords=ax.transAxes,

                size = 20

               )
# Create the pairgrid object

# 플랏할 대상을 이 오브젝트에서 정의하고, 페어 플랏에서 대각선 , 대각선 위/아래 방향으로 원하는 플랏을 넣어줌

grid = sns.PairGrid(

    data = plot_data,

    size = 3,

    diag_sharey=False,

    hue='TARGET',

    vars = [x for x in list(plot_data.columns) if x != 'TARGET']

)

# 여기 까지 하면 그래프를 그릴 수 있는 자리만 나옴!



# Upper is a scatter plot

grid.map_upper(plt.scatter, alpha = 0.2)



# Diagonal is a histogram

grid.map_diag(sns.kdeplot)



# Bottom is density plot

grid.map_lower(sns.kdeplot, cmap=plt.cm.OrRd_r);



plt.suptitle('Ext Source and Age Features Pairs Plot', size=32, y=1.05)
# Make a new dataframe for polynomial features

poly_features = app_train[

    ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH','TARGET']]

poly_features_test = app_test[

    ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH']]



# imputer for handling missing values

from sklearn.preprocessing import Imputer

imputer = Imputer(strategy='median')



poly_target = poly_features['TARGET']

poly_features = poly_features.drop(columns=['TARGET'])



poly_features.head()
# Need to impute missing values

poly_features = imputer.fit_transform(poly_features)

poly_features_test = imputer.transform(poly_features_test)
poly_features
# Create the polynomial object with specified degree

from sklearn.preprocessing import PolynomialFeatures



poly_transformer = PolynomialFeatures(degree=3)
# Train the polynomial features

poly_transformer.fit(poly_features)



# Transform the features

poly_features = poly_transformer.transform(poly_features)

poly_features_test = poly_transformer.transform(poly_features_test)

print('Polynomial Features shape : ', poly_features.shape)
# 35개 변수 모두 출력함

poly_transformer.get_feature_names(input_features=[

    'EXT_SOURCE_1',

    'EXT_SOURCE_2',

    'EXT_SOURCE_3',

    'DAYS_BIRTH'

])
# Create a dataframe of the features

poly_features = pd.DataFrame(poly_features, 

                             columns=poly_transformer.get_feature_names(input_features=[

                                 'EXT_SOURCE_1',

                                 'EXT_SOURCE_2',

                                 'EXT_SOURCE_3',

                                 'DAYS_BIRTH',

                             ]))



# Add in the target

# 타겟값은 원래 이 데이터프레임에 있었는데, polynomial feature 만드느라고, 드랍시키고, 따로 새로운 변수에 저장중이였다.

# 그 값을 다시 그대로 가져온다.

poly_features['TARGET'] = poly_target
poly_features.head()
# Find the correlations with the target

poly_corrs = poly_features.corr()['TARGET'].sort_values()



# Display most negative and most positive

print(poly_corrs.head(10))

print(poly_corrs.tail(5))
# Put test features into dataframe

print('Polynomial Features shape for test data : ', poly_features_test.shape)
poly_features_test = pd.DataFrame(poly_features_test, 

                                  columns=poly_transformer.get_feature_names(input_features=[

                                      'EXT_SOURCE_1',

                                      'EXT_SOURCE_2',

                                      'EXT_SOURCE_3',

                                      'DAYS_BIRTH']))
poly_features_test.head()
# Merge polynomial featueres into training dataframe

poly_features['SK_ID_CURR'] = app_train['SK_ID_CURR']

app_train_poly = app_train.merge(poly_features, on='SK_ID_CURR', how='left')



# Merge polynomial features into testing dataframe

poly_features_test['SK_ID_CURR'] = app_test['SK_ID_CURR']

app_test_poly = app_test.merge(poly_features_test, on='SK_ID_CURR', how='left')
# Align the dataframes

app_train_poly, app_test_poly = app_train_poly.align(

    app_test_poly,

    join='inner',

    axis=1

)



# Print out the new shapes

print('Training data with polynomial features shape : ', app_train_poly.shape)

print('Testing data with polynomial features shape : ', app_test_poly.shape)
for item in app_train_poly.columns:

    print(item)
# 새로운 컬럼을 만드는 과정, 기존에 있었던 컬럼들을 조합해서 유용한 정보를 가지는 텀들을 생성

# annuity : 연금

app_train_domain = app_train.copy()

app_test_domain = app_test.copy()



app_train_domain['CREDIT_INCOME_PERCENT'] = app_train_domain['AMT_CREDIT'] / app_train_domain['AMT_INCOME_TOTAL']

app_train_domain['ANNUITY_INCOME_PERCENT'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_INCOME_TOTAL']

app_train_domain['CREDIT_TERM'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_CREDIT']

app_train_domain['DAYS_EMPLOYED_PERCENT'] = app_train_domain['DAYS_EMPLOYED'] / app_train_domain['DAYS_BIRTH']
app_train_domain.head()
app_test_domain['CREDIT_INCOME_PERCENT'] = app_test_domain['AMT_CREDIT'] / app_test_domain['AMT_INCOME_TOTAL']

app_test_domain['ANNUITY_INCOME_PERCENT'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_INCOME_TOTAL']

app_test_domain['CREDIT_TERM'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_CREDIT']

app_test_domain['DAYS_EMPLOYED_PERCENT'] = app_test_domain['DAYS_EMPLOYED'] / app_test_domain['DAYS_BIRTH']
app_test_domain.head()
plt.figure(figsize=(12,20))

# iterate through the new features

for i, feature in enumerate(['CREDIT_INCOME_PERCENT','ANNUITY_INCOME_PERCENT','CREDIT_TERM','DAYS_EMPLOYED_PERCENT']):

    

    # create a new subplot for each source

    plt.subplot(4,1,i+1)

    # plot repaid loans

    sns.kdeplot(app_train_domain.loc[app_train_domain['TARGET'] == 0, feature], label='target == 0')

    # plot loans that were not repaid

    sns.kdeplot(app_train_domain.loc[app_train_domain['TARGET'] == 1, feature], label='target == 1')

    

    # Label the plots

    plt.title('Distribution of %s by Target Value' % feature)

    plt.xlabel('%s' % feature); plt.ylabel('Density');



plt.tight_layout(h_pad = 2.5)

    
from sklearn.preprocessing import MinMaxScaler, Imputer



# Drop the target from the training data

if 'TARGET' in app_train:

    train = app_train.drop(columns=['TARGET'])

else:

    train = app_train.copy()



# Feature names

features = list(train.columns)



# Copy of the testing data

test = app_test.copy()
# Median imputation of missing values

imputer = Imputer(strategy = 'median')



# Scale each feature to 0-1

scaler = MinMaxScaler(feature_range=(0,1))



# Fit on the training data

# fit 과정은 Imputer의 strategy를 `median`으로 했기 때문에, 

# 트레이닝 데이터를 기준으로 중간값? 을 찾아서 객체에 가지고 있을 것으로 유추할수? 있다.

imputer.fit(train)



# Transform both training and testing data

train = imputer.transform(train)

test = imputer.transform(app_test)

# 여기서 왠지모르겠지만 테스트 데이터는 원래 데이터를 그대로 사용? 

# 위에 먼저 만든 test 라는 값도 어차피 카피 한 것이기 때문에 값은 같겠지만, 튜토리얼에서 하는대로 일단 한다.

# 이렇게 missing value를 채워준다.



# Repeat with the scaler(0~1)

scaler.fit(train)

train = scaler.transform(train)

test = scaler.transform(test)



print('Training data shape : ', train.shape)

print('Testing data shape : ', test.shape)



from sklearn.linear_model import LogisticRegression



# Make the model with the specified regularization parameter

log_reg = LogisticRegression(C=0.0001)



# Train on the training data

# train_labels값은 app_train['TARGET'] 을 할당한 변수임

log_reg.fit(train, train_labels)

# (X, y) 형태로 데이터를 핏팅하고, 해당 모델이 이진 분류 문제로 적용되기 때문에, 예측되는 결과도 두가지 종류에 대해서 확률 값으로 반환된다.
# Make predictions

# Make sure to select the second column only

# input 은 테스트 데이터, 보고 싶은 데이터는 모든 행에 대하여 두번째 컬럼이다.

log_reg_pred = log_reg.predict_proba(test)[:,1]
log_reg_pred
# Submission dataframe

submit = app_test[['SK_ID_CURR']]

submit['TARGET']=log_reg_pred



submit.head()
# Save the submission to a csv file.

submit.to_csv('log_reg_baseline.csv', index=False)
# 대출상환 가능/불가능에 대한 분류 문제 이므로, classifier를 선택!

from sklearn.ensemble import RandomForestClassifier



# Make random forest classifier

random_forest = RandomForestClassifier(

    n_estimators=100,

    random_state=50,

    verbose=1,

    n_jobs=-1

)

# 일단 에스티메이터? 랜덤포레스트 알고리즘을 가지는 객체를 생성한거임
# Train on the training data

random_forest.fit(train,train_labels)
# Extract feature importances

feature_importance_value = random_forest.feature_importances_ # np.ndarray 반환함

feature_importances = pd.DataFrame({'feature':features,'importance':feature_importance_value})
# Make predictions on test data

predictions = random_forest.predict_proba(test)[:,1]
# Make a submission dataframe

submit = app_test[['SK_ID_CURR']]

submit['TARGET'] = predictions



submit.head()
# Save the submission dataframe

submit.to_csv('random_forest_baseline.csv', index=False)
poly_features_names = list(app_train_poly.columns)



# Impute the polynomial features

imputer = Imputer(strategy = 'median')



poly_features = imputer.fit_transform(app_train_poly)

poly_features_test = imputer.transform(app_test_poly)



# Scale the polynomial features

scaler = MinMaxScaler(feature_range=(0,1))



poly_features = scaler.fit_transform(poly_features)

poly_features_test = scaler.transform(poly_features_test)



random_forest_poly = RandomForestClassifier(n_estimators=100, random_state=50, verbose=1, n_jobs=-1)
# Train on the training data

# 핏팅할때 훈련데이터의 타겟값, 즉 상환 여부에 대한 레이블 값은 어떤 피쳐를 사용하든, 알고리즘을 사용하든 같으므로,

# 여기서도 train_labels을 사용함.

random_forest_poly.fit(poly_features, train_labels)



# Make predictions on the test data

predictions = random_forest_poly.predict_proba(poly_features_test)[:,1]
predictions
# Make a submission dataframe

submit = app_test[['SK_ID_CURR']]

submit['TARGET'] = predictions



# Save the submission datafrmae

submit.to_csv('random_forest_baseline_engineered.csv', index=False)

app_train_domain.head()
# 입력데이터에서 타겟 컬럼을 분리한다.

app_train_domain = app_train_domain.drop(columns='TARGET')



# 도메인 피쳐 이름 생성, 리스트

domain_features_names = list(app_train_domain.columns)



# Impute the domain nomial features

imputer = Imputer(strategy = 'median')



domain_features = imputer.fit_transform(app_train_domain)

domain_features_test = imputer.transform(app_test_domain)



# Scale the domainnomial features

scaler = MinMaxScaler(feature_range=(0,1))



domain_features = scaler.fit_transform(domain_features)

domain_features_test = scaler.transform(domain_features_test)



# Create Random Forest object for domain features data

random_forest_domain = RandomForestClassifier(n_estimators=100, random_state=50, verbose=1, n_jobs=-1)



# Train on the training data

random_forest_domain.fit(domain_features, train_labels)



# Extract feature importances, 후 작업을 위한 사전 작업

feature_importance_values_domain = random_forest_domain.feature_importances_

feature_importance_domain = pd.DataFrame({'feature':domain_features_names,'importance':feature_importance_values_domain})



# Make prediction on the test data

predictons = random_forest_domain.predict_proba(domain_features_test)[:,1]









# Make a submission dataframe

submit = app_test[['SK_ID_CURR']]

submit['TARGET'] = predictions



# Save the submission dataframe

submit.to_csv('random_forest_baseline_domain.csv', index=False)
#영향계수? 를 플랏하는 함수를 작성!

def plot_feature_importances(df):

    """

    Plot importances returned by a model. This can work with any measure of 

    feature importance provided that higher importance is better.

    

    Args:

        df(dataframe): feature importances. Must have the features in a column

        called 'features' and the importances in a column called 'importance'

        

    Reutrns:

        shows a plot of the 15 most importance features

        

        df(dataframe): feature importances sorted by importance(hightest to lowest)

        with a column for normalized importance

    """

    

    # Sort features according to importance

    # reset_index()를 하면, sorting 후 내림차순이 된 데이터의 row에 대해서 다시 0부터 인덱스를 매겨준다!

    df = df.sort_values('importance', ascending = False).reset_index()

    

    # Normalize the feature importances to add up to one

    # 전체 영향계수? 를 더하면 1이 되므로, 그 값으로 각각의 영향 계수를 나눠서 일반화 해줌.(0,1)사이의 값을 갖는다.

    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    

    # Make a horizontal bar chart of feature importances

    plt.figure(figsize=(10,6))

    ax = plt.subplot()

    

    # Need to reverse the index to plot most important on top

    # 여기서 인덱스를 리버스 하면서 앞에 list를 겹처서 붙여주는 이유는, 

    # 상속되는 대상이 객체(object) 타입의? 데이터라서, 리스트 화 시켜주기 위함이다.

    ax.barh(list(reversed(list(df.index[:15]))), 

            df['importance_normalized'].head(15), 

            align='center', 

            edgecolor='k')

    

    # Set the yticks and labels, y축 값 범위 정해주고, y축 데이터 라벨 이름 할당해주기

    ax.set_yticks(list(reversed(list(df.index[:15]))))

    ax.set_yticklabels(df['feature'].head(15))

    

    # Plot labeling

    plt.xlabel('Normalized Importance')

    plt.title('Feature Importances')

    plt.show()

    

    return df
# Show the feature importances for the dafault features

feature_importances_sorted = plot_feature_importances(feature_importances)
feature_importances_domain_sorted = plot_feature_importances(feature_importance_domain)
# 필요한 모듈 불러오기

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

import lightgbm as lgb

import gc
def model(features, test_features, encoding = 'ohe', n_folds = 5):

    """

    Train and Test a light gradient boosting model using cross validation.

    

    Parameters:

    -----------

        features(pd.DataFrame):

            dataframe of training feature to use

            for training a model. Must include Target column.

        test_features(pd.DataFrame):

            dataframe of testing feature to use

            for making predictions with the model.

        encoding(str, default = 'ohe'):

            method for encoding categorical variables. Either 'ohe' for one-hot encoding or 'le' for interger label encoding

        n_folds(int, default = 5): number of folds to use for cross validation

        

    Return:

    -------

        submission(pd.DataFrame):

            dataframe with 'SK_ID_CURR' and 'TARGET' probabilities

            predicted by the model.

        feature_importances(pd.DataFrame):

            dataframe with the feature importances from the model.

        valid_metrics(pd.DataFrame):

            dataframe with training and validation metrics(ROC AUC) for each fold and overall.

    """

    

    # Extract the ids

    train_ids = features['SK_ID_CURR']

    test_ids = test_features['SK_ID_CURR']

    

    # Extract the labels for training

    labels = features['TARGET']

    

    # Remove the ids and target

    features = features.drop(columns = ['SK_ID_CURR', 'TARGET'])

    test_features = test_features.drop(columns = ['SK_ID_CURR'])

    

    

    # One Hot Encoding

    if encoding == 'ohe':

        features = pd.get_dummies(features)

        test_features = pd.get_dummies(test_features)

        

        # Align the dataframes by the columns

        features, test_features = features.align(test_features, join = 'inner', axis = 1)

        

        # No categorical indices to record

        cat_indices = 'auto'

    

    # Integer label encoding

    elif encoding == 'le':

        

        # Create a label encoder

        label_encoder = LabelEncoder()

        

        # List for storing categorical indices

        cat_indices = []

        

        # Iterate through each column

        for i, col in enumerate(features):

            if features[col].dtype == 'object':

                # Map the categorical features to integers

                features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))

                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))



                # Record the categorical indices

                cat_indices.append(i)

    

    # Catch error if label encoding scheme is not valid

    else:

        raise ValueError("Encoding must be either 'ohe' or 'le'")

        

    print('Training Data Shape: ', features.shape)

    print('Testing Data Shape: ', test_features.shape)

    

    # Extract feature names

    feature_names = list(features.columns)

    

    # Convert to np arrays

    features = np.array(features)

    test_features = np.array(test_features)

    

    # Create the kfold object

    k_fold = KFold(n_splits = n_folds, shuffle = True, random_state = 50)

    

    # Empty array for feature importances

    feature_importance_values = np.zeros(len(feature_names))

    

    # Empty array for test predictions

    test_predictions = np.zeros(test_features.shape[0])

    

    # Empty array for out of fold validation predictions

    out_of_fold = np.zeros(features.shape[0])

    

    # Lists for recording validation and training scores

    valid_scores = []

    train_scores = []

    

    # Iterate through each fold

    for train_indices, valid_indices in k_fold.split(features):

        

        # Training data for the fold

        train_features, train_labels = features[train_indices], labels[train_indices]

        # Validation data for the fold

        valid_features, valid_labels = features[valid_indices], labels[valid_indices]

        

        # Create the model

        model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', 

                                   class_weight = 'balanced', learning_rate = 0.05, 

                                   reg_alpha = 0.1, reg_lambda = 0.1, 

                                   subsample = 0.8, n_jobs = -1, random_state = 50)

        

        # Train the model

        model.fit(train_features, train_labels, eval_metric = 'auc',

                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],

                  eval_names = ['valid', 'train'], categorical_feature = cat_indices,

                  early_stopping_rounds = 100, verbose = 200)

        

        # Record the best iteration

        best_iteration = model.best_iteration_

        

        # Record the feature importances

        feature_importance_values += model.feature_importances_ / k_fold.n_splits

        

        # Make predictions

        test_predictions += model.predict_proba(test_features, num_iteration = best_iteration)[:, 1] / k_fold.n_splits

        

        # Record the out of fold predictions

        out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]

        

        # Record the best score

        valid_score = model.best_score_['valid']['auc']

        train_score = model.best_score_['train']['auc']

        

        valid_scores.append(valid_score)

        train_scores.append(train_score)

        

        # Clean up memory

        gc.enable()

        del model, train_features, valid_features

        gc.collect()

        

    # Make the submission dataframe

    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})

    

    # Make the feature importance dataframe

    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

    

    # Overall validation score

    valid_auc = roc_auc_score(labels, out_of_fold)

    

    # Add the overall scores to the metrics

    valid_scores.append(valid_auc)

    train_scores.append(np.mean(train_scores))

    

    # Needed for creating dataframe of validation scores

    fold_names = list(range(n_folds))

    fold_names.append('overall')

    

    # Dataframe of validation scores

    metrics = pd.DataFrame({'fold': fold_names,

                            'train': train_scores,

                            'valid': valid_scores}) 

    

    return submission, feature_importances, metrics

    
submission, fi, metrics = model(app_train, app_test)

print('Baseline metrics')

print(metrics)
fi_sorted = plot_feature_importances(fi)
submission.to_csv('baseline_lgb.csv', index=False)
app_train_domain['TARGET'] = train_labels



# Test the domain knowledge features

sumission_domain, fi_domain, metrics_domain = model(

    app_train_domain,

    app_test_domain,

)



print('Baseline with domain knowledge features metrics')

print(metrics_domain)
fi_sorted = plot_feature_importances(fi_domain)
sumission_domain.to_csv('baseline_lgb_domain_features.csv', index=False)