import numpy as np

import pandas as pd



import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        file_path = dirname +'/'+ filename



# Read file

df = pd.read_csv(file_path)



df.head()
df.isnull().sum()
df = df.drop(

        ['job_id', 'title', 'location', 'department', 'salary_range', 'company_profile', 'description', 'requirements',

         'benefits'], axis=1).sort_index()
y = df['fraudulent']



y
X = df.drop(['fraudulent'], axis=1)



X.head()
# Different features for my model

numerical_features = ['telecommuting', 'has_company_logo', 'has_questions']

label_features = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']

for feature in label_features:

    X[feature].replace(np.nan, X[feature].mode()[0], regex=True, inplace=True)

    

X.head()
'''

    numeric_transformer = Pipeline(steps=[

        ('imputer', SimpleImputer(strategy='median'))])



    categorical_transformer = Pipeline(steps=[

        ('cat_imputer', OneHotEncoder())])



    preprocessing = ColumnTransformer(transformers=[

        ('numerical', numeric_transformer, numerical_features),

        ('categorical', categorical_transformer, label_features)

    ])



    log_reg = Pipeline(steps=[

        ('preprocessing', preprocessing),

        ('scaler', StandardScaler(with_mean=False)),

        ('log', LogisticRegression())

    ])

    '''
from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer, make_column_transformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import mean_absolute_error

from imblearn.under_sampling import RandomUnderSampler

from sklearn.decomposition import PCA
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')



imputer.fit_transform(X[numerical_features])



c_t = make_column_transformer((OneHotEncoder(), label_features), remainder='passthrough')



big_X = c_t.fit_transform(X).toarray()



sc = StandardScaler()



big_X = sc.fit_transform(big_X)



pca = PCA()



big_X = pca.fit_transform(big_X)



rus = RandomUnderSampler()



undersampled_x, y = rus.fit_resample(big_X,y)



x_train, x_test, y_train, y_test = train_test_split(undersampled_x, y, test_size=0.2, random_state=0)



log_reg = LogisticRegression()



log_reg.fit(x_train, y_train)



y_pred = log_reg.predict(x_test)



print(confusion_matrix(y_test, y_pred))

print(f'Prediction score: {log_reg.score(x_test, y_test) * 100:.2f}%')

print(f'MAE from Logistic Regression: {mean_absolute_error(y_test, y_pred) * 100:.2f}%')
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV
knn = KNeighborsClassifier()



grid = GridSearchCV(knn, param_grid={'n_neighbors':range(1,31)}, scoring='accuracy')



grid.fit(undersampled_x,y)



for i in range(0, len(grid.cv_results_['mean_test_score'])):

    print('N_Neighbors {}: {} '.format(i+1, grid.cv_results_['mean_test_score'][i]*100))
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(bootstrap=True)



rf.fit(undersampled_x,y)



# Cross validation of 5 folds

score = cross_val_score(rf, undersampled_x, y)



print(f'Prediction score: {np.mean(score) * 100:.2f}%')