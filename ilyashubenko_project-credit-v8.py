import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Графика
import matplotlib.pyplot as plt
import seaborn as sns

# Отображение в ноут
%matplotlib inline

# Сплит, метрики
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV, learning_curve
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

# Модель
import catboost as catb

# Файлы
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
TEST_DATASET_FILE = '../input/credit-default/test.csv'
TRAIN_DATASET_FILE = '../input/credit-default/train.csv'
SAMPLE = '../input/credit-default/sample_submission.csv'

df_train = pd.read_csv(TRAIN_DATASET_FILE)
df_test = pd.read_csv(TEST_DATASET_FILE)

TARGET_NAME = 'Credit Default'
BASE_FEATURE_NAMES = df_train.columns.drop(TARGET_NAME).tolist()
def get_classification_report(y_train_true, y_train_pred, y_test_true, y_test_pred):
    print('TRAIN\n\n' + classification_report(y_train_true, y_train_pred))
    print('TEST\n\n' + classification_report(y_test_true, y_test_pred))
    print('CONFUSION MATRIX\n')
    print(pd.crosstab(y_test_true, y_test_pred))
    
def show_feature_importances(feature_names, feature_importances, get_top=None):
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
    feature_importances = feature_importances.sort_values('importance', ascending=False)
    
    plt.figure(figsize = (10, 6))
    
    sns.barplot(feature_importances['importance'], feature_importances['feature'])
    
    plt.xlabel('Importance')
    plt.title('Importance of features')
    plt.show()
    
    if get_top is not None:
        return feature_importances['feature'][:get_top].tolist()
    
def show_learning_curve_plot(estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, 
                                                            cv=cv, 
                                                            scoring='f1',
                                                            train_sizes=train_sizes, 
                                                            n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(15,8))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.title(f"Learning curves ({type(estimator).__name__})")
    plt.xlabel("Training examples")
    plt.ylabel("Score")     
    plt.legend(loc="best")
    plt.grid()
    plt.show()
    
# Целевая переменная
y = df_train[[TARGET_NAME]]
y.info()
plt.figure(figsize=(8, 5))

sns.countplot(x=TARGET_NAME, data=df_train)

plt.title('Target variable distribution')
plt.show()
df_train.head(10)
corr_with_target = df_train.corr().iloc[:-1, -1].sort_values(ascending=False)

plt.figure(figsize=(10, 8))

sns.barplot(x=corr_with_target.values, y=corr_with_target.index)

plt.title('Correlation with target variable')
plt.show()
plt.figure(figsize=(10, 8))

sns.countplot(x="Years in current job", hue=TARGET_NAME, data=df_train)
plt.title('\"Years in current job\" grouped by target variable')
plt.legend(title='Target', loc='upper right')

plt.show()
# пропуски
df_train.isna().sum()
for cat_colname in df_train.select_dtypes(include='object').columns:
    print(str(cat_colname) + '\n\n' + str(df_train[cat_colname].value_counts()) + '\n' + '*' * 100 + '\n')
df_train['Term'].value_counts()
def prepare_usage(data, train = True):
    
    # Пропуски и 0 заменяем на средние
    mean_annual_income = data[~((data['Annual Income'].isna()) | (data['Annual Income'] == 0))]['Annual Income'].mean()
    data.loc[(data['Annual Income'].isna()), ['Annual Income']] = mean_annual_income
    
    median_credit_score = data[~((data['Credit Score'].isna()) | (data['Credit Score'] == 0))]['Credit Score'].mean()
    data.loc[(data['Credit Score'].isna()), ['Credit Score']] = median_credit_score
    
    median_months_since_last_delinquent = data[~((data['Months since last delinquent'].isna()) | 
                                                 (data['Months since last delinquent'] == 0))]['Months since last delinquent'].mean()
    data.loc[(data['Months since last delinquent'].isna()), ['Months since last delinquent']] = median_months_since_last_delinquent
    
    #Удаляем строки где целевая переменная = 0 и есть значения Nan
    columns_to_drop_NAN = ['Years in current job', 'Bankruptcies',]
    for col in columns_to_drop_NAN:
        if train:
            data = data[~((data[col].isna()) & (data[TARGET_NAME] == 0))]
        else:
            pass
    
    # Home Ownership
    map_home_ownership = {
        "Have Mortgage" : 0,
        "Rent" : 1,
        "Own Home" : 2,
        "Home Mortgage" : 3
    }
    
    data["Home Ownership"] = data["Home Ownership"].map(map_home_ownership)
    
    # Years in current job
    map_years_in_current_job = {
        "< 1 year" : 0,
        "1 year" : 1,
        "2 years" : 2,
        "3 years" : 3,
        "4 years" : 4,
        "5 years" : 5,
        "6 years" : 6,
        "7 years" : 7,
        "8 years" : 8,
        "9 years" : 9,
        "10+ years" : 10
        
    }
    
    data["Years in current job"] = data["Years in current job"].map(map_years_in_current_job)
    
    # Purpose
    map_purpose = {
        "renewable energy" : 14,
        "vacation" : 13,
        "educational expenses" : 12,
        "moving" : 11,
        "wedding" : 10,
        "small business" : 9,
        "buy house" : 8,
        "take a trip" : 7,
        "major purchase" : 6,
        "medical bills" : 5,
        "buy a car" : 4,
        "business loan" : 3,
        "home improvements" : 2,
        "other" : 1,
        "debt consolidation" : 0
    }
    
    data["Purpose"] = data["Purpose"].map(map_purpose)
    
    # Term
    map_term = {
        "Short Term" : 0,
        "Long Term" : 1
    }
    
    data["Term"] = data["Term"].map(map_term)
    
    # Все оставшиеся NaN заменяем на 0
    data = data.fillna(0)
    
    return data
df_train
df_train = prepare_usage(df_train)
df_train
df_train.isnull().sum()
plt.figure(figsize=(8, 5))

sns.countplot(x = TARGET_NAME, data = df_train)

plt.title('Target variable distribution')
plt.show()
corr_with_target = df_train.corr().iloc[:-1, -1].sort_values(ascending=False)

plt.figure(figsize=(10, 8))

sns.barplot(x=corr_with_target.values, y=corr_with_target.index)

plt.title('Correlation with target variable')
plt.show()
plt.figure(figsize = (10, 3))

df_train['Credit Score'].hist(bins=30, )
plt.ylabel('Count')
plt.xlabel('Credit Score')

plt.title('credit score samples')
plt.show()
X = df_train[BASE_FEATURE_NAMES]
y = df_train[TARGET_NAME]
%%time
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=11)

cat_model = catb.CatBoostClassifier(
    auto_class_weights='Balanced',
    silent=True,
    #depth=3,
    #iterations=70,
    random_state=11,
    #l2_leaf_reg=10,
    )


#final_model = catb.CatBoostClassifier(n_estimators=500, max_depth=3, l2_leaf_reg=10,
#                                      silent=True, random_state=11)
cat_model.fit(X_train, y_train)

y_train_pred = cat_model.predict(X_train)
y_test_pred = cat_model.predict(X_test)

get_classification_report(y_train, y_train_pred, y_test.values, y_test_pred)
important_features_top = show_feature_importances(X_train.columns, cat_model.feature_importances_, get_top=15)
params = {'n_estimators':[50, 100, 200, 500, 700, 1000, 1200, 1500],
          'max_depth':[3, 5, 7]}
cv=KFold(n_splits=3, random_state=11, shuffle=True)
%%time

rs = RandomizedSearchCV(cat_model, params, scoring='f1', cv=cv, n_jobs=-1)
rs.fit(X, y)
rs.best_params_
rs.best_score_
%%time
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.28, random_state=11)

final_model = catb.CatBoostClassifier(
    auto_class_weights='Balanced',
    silent=True,
    depth=3,
    #n_estimators=200,
    iterations=1500,
    #l2_leaf_reg=10,
    random_state=11,
    reg_lambda=0.8)

final_model.fit(X_train, y_train)

y_train_pred = final_model.predict(X_train)
y_test_pred = final_model.predict(X_test)

get_classification_report(y_train, y_train_pred, y_test.values, y_test_pred)
show_learning_curve_plot(final_model, X_train, y_train)
df_test.head(5)
df_test = prepare_usage(df_test, False)
df_test.head(15)
df_sample = pd.read_csv(SAMPLE)
df_sample
predictions = pd.DataFrame(index = df_sample['Id'])
y_pred_output = final_model.predict(df_test)
predictions[TARGET_NAME] = y_pred_output
predictions.to_csv('out_credit_12.csv', sep=',')
predictions.head(10)