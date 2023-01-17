import category_encoders as ce

from sklearn.pipeline import make_pipeline

from sklearn.tree import DecisionTreeClassifier

import category_encoders as ce

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
df = pd.read_csv('../input/train.csv')
df.shape
df.columns
#statistics summary

df['SalePrice'].describe()
df = df.dropna(subset=['BedroomAbvGr'])

df['Great'] = df['BedroomAbvGr'] >= 4

cardinality = df.select_dtypes(exclude='number').nunique()



high_cardinality_feat = cardinality[cardinality > 20].index.tolist()

df = df.drop(columns = high_cardinality_feat)

df = df.fillna('Missing')



train = df[df['YrSold'] <= 2016]

val = df[df['YrSold'] == 2007]

test = df[df['YrSold'] <= 2008]



target = 'Great'

features = train.columns.drop([target, 'YrSold'])

X_train = train[features]

y_train = train[target]

X_val = val[features]

y_val = val[target]



pipeline = make_pipeline(

    ce.OrdinalEncoder(), 

    SimpleImputer(strategy='median'), 

    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

)

pipeline.fit(X_train, y_train)

print(f'Validation accuracy: {pipeline.score(X_val, y_val)}')
#scatter plot grlivarea/saleprice

var = 'GrLivArea'

data = pd.concat([df['SalePrice'], df[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#scatter plot totalbsmtsf/saleprice

var = 'TotalBsmtSF'

data = pd.concat([df['SalePrice'], df[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
import category_encoders as ce

from sklearn.linear_model import LinearRegression

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler



lr = make_pipeline(

    ce.TargetEncoder(),  

    LinearRegression()

)



lr.fit(X_train, y_train)

print('Linear Regression R^2', lr.score(X_val, y_val))
from sklearn.metrics import r2_score

from xgboost import XGBRegressor



gb = make_pipeline(

    ce.OrdinalEncoder(), 

    XGBRegressor(n_estimators=200, objective='reg:squarederror', n_jobs=-1)

)



gb.fit(X_train, y_train)

y_pred = gb.predict(X_val)

print('Gradient Boosting R^2', r2_score(y_val, y_pred))
from pdpbox.pdp import pdp_interact, pdp_interact_plot

features = ['LotArea', 'GarageCars']



interact = pdp_interact(

    model=gb,

    dataset=X_val,

    model_features=X_val.columns,

    features=features

)
pdp = interact.pdp.pivot_table(

    values='preds', 

    columns=features[0], 

    index=features[1]

)[::-1]
import plotly.graph_objs as go



surface = go.Surface(

    x=pdp.columns, 

    y=pdp.index, 

    z=pdp.values

)





layout = go.Layout(

    scene=dict(

        xaxis=dict(title=features[0]), 

        yaxis=dict(title=features[1]), 

        zaxis=dict(title=target)

    )

)



fig = go.Figure(surface, layout)

fig.show()
import category_encoders as ce

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier



target = 'Great'

features = df.columns.drop(['Great', 'SaleCondition'])



X = df[features]

y = df[target]



# Use Ordinal Encoder, outside of a pipeline

encoder = ce.OrdinalEncoder()

X_encoded = encoder.fit_transform(X)



model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

model.fit(X_encoded, y)
feature = 'HouseStyle'

for item in encoder.mapping:

    if item['col'] == feature:

        feature_mapping = item['mapping']

        

feature_mapping = feature_mapping[feature_mapping.index.dropna()]

category_names = feature_mapping.index.tolist()

category_codes = feature_mapping.values.tolist()
features = ['HouseStyle', 'Fireplaces']



interaction = pdp_interact(

    model=model, 

    dataset=X_encoded, 

    model_features=X_encoded.columns, 

    features=features

)



pdp_interact_plot(interaction, plot_type='grid', feature_names=features);
pdp = interaction.pdp.pivot_table(

    values='preds', 

    columns=features[0], # First feature on x axis

    index=features[1]    # Next feature on y axis

)[::-1]  # Reverse the index order so y axis is ascending



pdp = pdp.rename(columns=dict(zip(category_codes, category_names)))

plt.figure(figsize=(10,8))

sns.heatmap(pdp, annot=True, fmt='.3f', cmap='viridis')

plt.title('House Style with fireplace');
#shap

import category_encoders as ce

from sklearn.pipeline import make_pipeline

from sklearn.tree import DecisionTreeClassifier

import category_encoders as ce

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split





df = df.dropna(subset=['BedroomAbvGr'])

df['Great'] = df['BedroomAbvGr'] >= 4



cardinality = df.select_dtypes(exclude='number').nunique()



high_cardinality_feat = cardinality[cardinality > 20].index.tolist()

df = df.drop(columns = high_cardinality_feat)

df = df.fillna('Missing')



train = df[df['YrSold'] <= 2016]



test = df[df['YrSold'] <= 2008]



# Assign to X, y

target = 'SalePrice'

features = ['GarageCars', 'Fireplaces', 'FullBath']

X_train = train[features]

y_train = train[target]

X_test = test[features]

y_test = test[target]
from scipy.stats import randint, uniform

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV



param_distributions = { 

    'n_estimators': randint(50, 500), 

    'max_depth': [5, 10, 15, 20, None], 

    'max_features': uniform(0, 1), 

}



search = RandomizedSearchCV(

    RandomForestRegressor(random_state=42), 

    param_distributions=param_distributions, 

    n_iter=5, 

    cv=2, 

    scoring='neg_mean_absolute_error', 

    verbose=10, 

    return_train_score=True, 

    n_jobs=-1, 

    random_state=42

)



search.fit(X_train, y_train);
print('Best hyperparameters', search.best_params_)

print('Cross-validation MAE', -search.best_score_)

model = search.best_estimator_
X_test.head(5)
X_test.shape
row = X_test.iloc[[0]]

row
y_test.iloc[0]
model.predict(row)
import shap



explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(row)

shap_values
explainer.expected_value
y_train.mean()
shap.initjs()

shap.force_plot(

    base_value=explainer.expected_value,

    shap_values=shap_values,

    features=row

)
def predict(GarageCars, Fireplaces, FullBath):



    # Make dataframe from the inputs

    df = pd.DataFrame(

        data=[[GarageCars, Fireplaces, FullBath]], 

        columns=['GarageCars', 'Fireplaces', 'FullBath']

    )



    # Get the model's prediction

    pred = model.predict(df)[0]



    # Calculate shap values

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(df)



    # Get series with shap values, feature names, & feature values

    feature_names = df.columns

    feature_values = df.values[0]

    shaps = pd.Series(shap_values[0], zip(feature_names, feature_values))



    # Print results

    result = f'${pred:,.0f} estimated sale price. \n\n'

    result += f'Starting from baseline of ${explainer.expected_value} \n'

    result += shaps.to_string()

    print(result)





    # Show shapley values force plot

    shap.initjs()

    return shap.force_plot(

        base_value=explainer.expected_value, 

        shap_values=shap_values, 

        features=df

    )



predict(3, 2, 2)
predict(2, 2, 2)
predict(2, 1, 2)
predict(3, 2, 3)