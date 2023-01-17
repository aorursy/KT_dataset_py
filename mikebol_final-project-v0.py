import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, ShuffleSplit, cross_val_score, learning_curve
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

import catboost as catb

from scipy.stats import shapiro
from scipy.stats import probplot
from scipy.stats import mannwhitneyu
from scipy.stats import chi2_contingency

import warnings
warnings.simplefilter('ignore')
TRAIN_DATASET_PATH = '../gb_2nd_project/train.csv'
TEST_DATASET_PATH = '../gb_2nd_project/test.csv'
train = pd.read_csv(TRAIN_DATASET_PATH) # загружаем тренировочный датасет в датафрейм df_train
test = pd.read_csv(TEST_DATASET_PATH) # загружаем тестовый датасет в датафрейм df_test
TARGET_NAME = 'Credit Default'
BASE_FEATURE_NAMES = train.columns.drop(TARGET_NAME).tolist()

NUMB_FEATURE_NAMES = ['Annual Income','Tax Liens','Number of Open Accounts','Years of Credit History',
                     'Maximum Open Credit','Number of Credit Problems','Months since last delinquent',
                     'Bankruptcies','Current Loan Amount','Current Credit Balance','Monthly Debt','Credit Score']

CAT_FEATURE_NAMES = ['Years in current job','Home Ownership','Purpose','Term']
train.head(10)
plt.figure(figsize=(8, 5))

sns.countplot(x=TARGET_NAME, data=train)

plt.title('Target variable distribution')
plt.show()

plt.figure(figsize = (15,10))

sns.set(font_scale=0.8)

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
corr_matrix = train.corr()
corr_matrix = np.round(corr_matrix, 2)
corr_matrix[np.abs(corr_matrix) < 0.3] = 0

sns.heatmap(corr_matrix,mask=mask,  annot=True, linewidths=.5, cmap='coolwarm')

plt.title('Correlation matrix')
plt.show()
plt.figure(figsize=(10, 8))

sns.countplot(x="Years in current job", hue=TARGET_NAME, data=train)
plt.title('\"Years in current job\" grouped by target variable')
plt.legend(title='Target', loc='upper right')

plt.show()
train.isna().sum()
class FeatureCorrector:
    
    def __init__(self):
        
        self.maximum_open_credit_max = None
        self.current_loan_amount_max = None
        self.medians = None
        self.year_mode = None
        self.mask_year = None
        self.mask_deliq = None
        self.mask_term = None
        
    def fit(self, df):
        
        self.max_open_credit_max = df['Maximum Open Credit'].quantile(0.85)
        self.cur_loan_amount_max = 800000 #df['Current Loan Amount'].quantile(0.9)
        self.medians = df.median()
        self.year_mode = df['Years in current job'].mode().astype(str)[0]
        self.mask_deliq = {True : 0, False: 1}
        self.mask_term = {'Short Term': 0, 'Long Term': 1}
        
        
    def transform(self, df):
        
        # Annual Income
        
        df['Annual_Income_is_NaN'] = 0
        df.loc[df['Annual Income'].isnull() == True, 'Annual_Income_is_NaN'] = 1
        
        df['Annual Income'].fillna(self.medians['Annual Income'], inplace=True)
        
        # Years in current job
        
        df['Years in current job'] = df['Years in current job'].fillna(self.year_mode)
        
        # Home Ownership
        
        df.loc[train["Home Ownership"] == "Have Mortgage", "Home Ownership"] = "Home Mortgage"
        
        # Tax Liens
        
        df.loc[df['Tax Liens'] > 0, 'Tax Liens'] = 1
        
        # Months since last delinquent
        
        df['Months since last delinquent'] = df['Months since last delinquent'].isna().map(self.mask_deliq)
        
        # Bankruptcies
        
        df["Bankruptcies"].fillna(0, inplace=True)
        
        # Term
        
        df['Term'] = df['Term'].map(self.mask_term)
        
        # Current Loan Amount
        
        df.loc[df['Current Loan Amount'] > self.cur_loan_amount_max , 'Current Loan Amount'] = self.cur_loan_amount_max
        df['Current Loan Amount'].fillna(self.medians['Current Loan Amount'])
        
        # Maximum Open Credit
        
        df.loc[df['Maximum Open Credit'] > self.max_open_credit_max, 'Maximum Open Credit'] = self.max_open_credit_max
        
        # Credit Score
        
        df.loc[df['Credit Score'] > 1000, 'Credit Score'] = df['Credit Score'] / 10
        df['Credit Score'].fillna(self.medians['Credit Score'], inplace=True)
        
        # convert to int64
        
        df['Tax Liens'] = df['Tax Liens'].astype('int64')
        df['Bankruptcies'] = df['Bankruptcies'].astype('int64')
        df['Number of Credit Problems'] = df['Number of Credit Problems'].astype('int64')
        
        return df
corrector = FeatureCorrector()

corrector.fit(train)

train = corrector.transform(train)
test = corrector.transform(test)
train.info()
train.isna().sum().sum()
test.isna().sum().sum()
def get_classification_report(y_train_true, y_train_pred, y_test_true, y_test_pred):
    print('TRAIN\n\n' + classification_report(y_train_true, y_train_pred))
    print('TEST\n\n' + classification_report(y_test_true, y_test_pred))
    print('CONFUSION MATRIX\n')
    print(pd.crosstab(y_test_true, y_test_pred))
def balance_df_by_target(df, target_name):

    target_counts = df[target_name].value_counts()

    major_class_name = target_counts.argmax()
    minor_class_name = target_counts.argmin()

    disbalance_coeff = int(target_counts[major_class_name] / target_counts[minor_class_name]) - 1

    for i in range(disbalance_coeff):
        sample = df[df[target_name] == minor_class_name].sample(target_counts[minor_class_name])
        df = df.append(sample, ignore_index=True)

    return df.sample(frac=1)
NEW_FEATURE_NAMES = ['Annual_Income_is_NaN']

corr_with_target = train[BASE_FEATURE_NAMES + 
                            NEW_FEATURE_NAMES +
                            [TARGET_NAME]].corr().iloc[:-1, -1].sort_values(ascending=False)

plt.figure(figsize=(10, 8))

sns.barplot(x=corr_with_target.values, y=corr_with_target.index)

plt.title('Correlation with target variable')
plt.show()

# Количество людей, разделенных по стажу, которые возращают и не возвращают кредит

g = sns.catplot("Credit Default", col="Years in current job", col_wrap=5,
                data=train,
                kind="count", height=3.5, aspect=.8, 
                palette='tab20')

plt.show()
X = train[BASE_FEATURE_NAMES]
y = train[TARGET_NAME]

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.30, random_state=11)
disbalance = y_train.value_counts()[0] / y_train.value_counts()[1]
disbalance
# выбор пал на catboost, потому что методом подбора моделей, учитывая категориальные признаки, лучше всего себя показывает
# потому что по дефолту имеет огромный спектр по подбору гиперпараметров
# я не приводил других моделей, потому что они себя не очень стабильно вели, опираясь на мою чистку данных

model_catb = catb.CatBoostClassifier(class_weights=[1, disbalance], silent=True, random_state=11, cat_features=CAT_FEATURE_NAMES)
model_catb.fit(X_train, y_train)

y_train_pred = model_catb.predict(X_train)
y_test_pred = model_catb.predict(X_test)

get_classification_report(y_train, y_train_pred, y_test, y_test_pred)
model_catb = catb.CatBoostClassifier(class_weights=[1, disbalance], silent=True, random_state=11, cat_features=CAT_FEATURE_NAMES)
params = {'n_estimators':[100, 200, 500, 700, 1000, 1200, 1500],
          'max_depth':[3, 5, 7]}
cv=KFold(n_splits=5, random_state=11, shuffle=True)
%%time

rs = RandomizedSearchCV(model_catb, params, scoring='f1', cv=cv, n_jobs=-1)
rs.fit(X, y)
rs.best_params_
rs.best_score_
%%time
# поэкспериментировав с полученными параметрами, лучшими были 700 и 3



final_model = catb.CatBoostClassifier(n_estimators=700, max_depth=3,
                                      class_weights=[1, disbalance],
                                      eval_metric='F1',
                                      cat_features=CAT_FEATURE_NAMES,
                                      silent=True, random_state=11)
final_model.fit(X_train, y_train)

y_train_pred = final_model.predict(X_train)
y_test_pred = final_model.predict(X_test)

get_classification_report(y_train, y_train_pred, y_test, y_test_pred)
df_model = test[BASE_FEATURE_NAMES]

y_pred = final_model.predict(df_model)

preds_final = pd.DataFrame()
preds_final = pd.DataFrame({'Id': np.arange(0,y_pred.shape[0]), 'Credit Default': y_pred})
preds_final
preds_final.to_csv('./predictions_new.csv', index=False, encoding='utf-8', sep=',')