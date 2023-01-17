import numpy as np
import pandas as pd
import json
from urllib.request import urlopen
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import calmap
import folium
from numpy import mean
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import eli5
from eli5.sklearn import PermutationImportance
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots
import shap 
full_table = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')
full_table.head()
full_table.info()
full_table.isna().sum()
cases = ['HDI for year']
full_table[cases] = full_table[cases].fillna(0)
plt.figure(figsize=(40,20))
bar_age = sns.barplot(x = 'sex', y = 'suicides_no', hue = 'age',data = full_table)
sns.set(font_scale=6)

sns.catplot(x="sex", y="suicides_no", col="country",

                data=full_table, kind="bar",

                height=4, aspect=1,col_wrap=3,estimator=mean);
sns.set(style="white",font_scale=1)
sns.catplot(x="age", y="suicides_no", col="country",

                data=full_table, kind="bar",

                height=3, aspect=4,col_wrap=1);
sns.set(style="darkgrid")

g = sns.FacetGrid(full_table, row="year", col="age",hue='sex',margin_titles=True)

(g.map(plt.scatter, "suicides_no","population", edgecolor="w")).add_legend()
full_table.pivot_table(index='country', columns='sex',values='suicides/100k pop',aggfunc='mean').plot(kind='barh',figsize=(100,300))
sns.set(style="white",font_scale=10)
sns.catplot("generation",

                data=full_table,

                kind="count", height=10, aspect=2)
sns.set(style="white",font_scale=3)
plt.figure(figsize=(80,50))
sns.regplot(x="gdp_per_capita ($)", y="suicides/100k pop", data=full_table)
sns.set(style="white",font_scale=7)
full_table.pivot_table(index='year',values='suicides_no', aggfunc='sum').plot(kind='line',figsize=(40,20))
sns.set(style="white",font_scale=6)
td=full_table.groupby(['country']).mean()
fd=td.reset_index()
concap = pd.read_csv('../input/concap1/concap.csv')
concap
suic_sum = pd.DataFrame(full_table['suicides_no'].groupby(full_table['country']).mean())
suic_sum = suic_sum.reset_index().sort_values(by='suicides_no',ascending=False)
def reg(x):
    if x=='Russia':
        res = 'Russian Federation'
    else:
        res=x
    return res
concap['CountryName'] = concap['CountryName'].apply(reg)
df = pd.merge(concap[['CountryName', 'CapitalName', 'CapitalLatitude', 'CapitalLongitude']],suic_sum,left_on='CountryName',right_on='country')
df
world_map = folium.Map(location=[10,0], tiles="cartodbpositron", zoom_start=2,max_zoom=6,min_zoom=2)
for i in range(0,len(df)):
    folium.Circle(
        location=[df.iloc[i]['CapitalLatitude'], df.iloc[i]['CapitalLongitude']],
        tooltip = "<h5 style='text-align:center;font-weight: bold'>"+df.iloc[i]['CountryName']+"</h5>"+
                    "<div style='text-align:center;'>"+"</div>"+
                    "<hr style='margin:10px;'>"+
                    "<ul style='color: #444;list-style-type:circle;align-item:left;padding-left:20px;padding-right:20px'>"+
        "<li>suicides_no: "+str(df.iloc[i,5])+"</li>"
        ,
        radius=(int((np.log(df.iloc[i,5]+1.00001)))+0.2)*50000,
        color='#ff0000',
        fill_color='#ff8533',
        fill=True).add_to(world_map)

world_map
plt.figure(figsize=(60,30))
cor = sns.heatmap(full_table.corr(), annot = True)
sns.set(font_scale=5)
cols_to_use = ['country', 'year', 'sex', 'age','population','gdp_per_capita ($)']
object_cols=['country','sex', 'age']
X = full_table[cols_to_use]
y = full_table.suicides_no
Xp = pd.read_csv('../input/usa2016/su.csv')
X_train, X_valid, y_train, y_valid = train_test_split(X, y)
label_encoder = LabelEncoder()
for col in object_cols:
    X_train[col] = label_encoder.fit_transform(X_train[col])
    X_valid[col] = label_encoder.transform(X_valid[col])
    Xp[col]=label_encoder.transform(Xp[col])

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.1,n_jobs=4)
my_model.fit(X_train, y_train)

predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))
Xp
predictions1 = my_model.predict(Xp)
predictions1
perm = PermutationImportance(my_model, random_state=1).fit(X_valid, y_valid)
eli5.show_weights(perm, feature_names = X_valid.columns.tolist())
pdp_goals = pdp.pdp_isolate(model=my_model, dataset=X_valid, model_features=cols_to_use, feature='population')


pdp.pdp_plot(pdp_goals, 'population')
plt.show()
pdp_goals = pdp.pdp_isolate(model=my_model, dataset=X_valid, model_features=cols_to_use, feature='gdp_per_capita ($)')

pdp.pdp_plot(pdp_goals, 'gdp_per_capita ($)')
plt.show()
data_for_prediction = X_valid.iloc[5]
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
explainer = shap.TreeExplainer(my_model)
shap_values = explainer.shap_values(data_for_prediction)
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values,  data_for_prediction)
explainer = shap.TreeExplainer(my_model)

shap_values = explainer.shap_values(X_valid,check_additivity=False)

shap.summary_plot(shap_values, X_valid)
explainer = shap.TreeExplainer(my_model)

shap_values = explainer.shap_values(X_valid,check_additivity=False)

shap.dependence_plot('gdp_per_capita ($)', shap_values, X_valid, interaction_index="population")

