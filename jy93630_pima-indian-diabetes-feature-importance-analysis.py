import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

diabetes = pd.read_csv("../input/diabetes.csv")

print(diabetes.columns)
diabetes.head()
columns = list(diabetes.columns.values)
features = [x for x in columns if x != 'Outcome']
sns.pairplot(diabetes, hue='Outcome', 
             x_vars=features, y_vars=features, height=2.5)
plt.show()
# Percentages of missing values 
imputation_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for feature in imputation_features :
    print(feature, round(len(diabetes[diabetes[feature] == 0]) / len(diabetes), 4))
column_means = diabetes[imputation_features].replace(0, np.NaN).mean(axis=0)
for feature in imputation_features : 
    diabetes.loc[diabetes[feature]== 0, feature] = column_means[feature]
columns = list(diabetes.columns.values)
features = [x for x in columns if x != 'Outcome']
sns.pairplot(diabetes, hue='Outcome', 
             x_vars=features, y_vars=features, height=2.5)
plt.show()
import statsmodels.api as sm
logit = sm.Logit(diabetes['Outcome'], diabetes[features]) 
result = logit.fit()

# Scipy error fixing..
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

print(result.summary())
print("Odds ratio")
print(np.exp(result.params))
sns.set_context("paper")
sns.set(font='serif')
sns.set_style("white", {
    "font.family": "serif",
    "font.serif": ["Times", "Palatino", "serif"]
})
oddsratio = np.exp(result.params)
plt.figure(figsize=(8,6))
n_features = 8
plt.barh(range(n_features), oddsratio, align='center')
plt.yticks(np.arange(n_features), features)
plt.xlabel("Odds ratio")
plt.ylabel("Feature")
plt.ylim(-1, n_features)
plt.xlim(0.7, 1.5)
plt.show()
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=5, random_state=0)
tree.fit(diabetes[features], diabetes['Outcome'])

print("Accuracy on training set: {:.3f}".format(tree.score(diabetes[features], diabetes['Outcome'])))
diabetes_features = [x for i,x in enumerate(diabetes.columns) if i !=8]
print("Decision Tree Feature importances:\n{}".format(tree.feature_importances_))
def plot_feature_importances_diabetes(model):
    plt.figure(figsize=(8,6))
    n_features = 8
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), diabetes_features)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)

plot_feature_importances_diabetes(tree)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_depth=3, n_estimators=100, random_state=0)
rf.fit(diabetes[features], diabetes['Outcome'])
print('Random Forest - Max Depth = 3')
print("Accuracy on training set: {:.3f}".format(rf.score(diabetes[features], diabetes['Outcome'])))
print('Random Forest Feature Importance')
plot_feature_importances_diabetes(rf)
from pandas import read_csv, DataFrame
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence

model = GradientBoostingClassifier(n_estimators=100, max_depth=4,learning_rate=0.1, loss='deviance',random_state=1)
model.fit(diabetes[features], diabetes['Outcome'])
importances = model.feature_importances_
plot_feature_importances_diabetes(model)
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,20)

features_index = [0, 1, 2, 3, 4, 5, 6, 7, (1,5)] # For 1, 5 th features, draw interaction plot
fig, axs = plot_partial_dependence(model, diabetes[features], features_index,feature_names=features,n_jobs=3, grid_resolution=50)
plt.figure()
plt.subplots_adjust(bottom=0.1, right=1.1, top=1.4) 
fig = plt.figure()
target_feature = (1, 5)
pdp, axes = partial_dependence(model, target_feature,X=diabetes[features], grid_resolution=50)
XX, YY = np.meshgrid(axes[0], axes[1])
Z = pdp[0].reshape(list(map(np.size, axes))).T
ax = Axes3D(fig)
surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1,cmap=plt.cm.BuPu, edgecolor='k')
ax.set_xlabel(diabetes_features[target_feature[0]])
ax.set_ylabel(diabetes_features[target_feature[1]])
ax.set_zlabel('Partial dependence')

plt.colorbar(surf)
plt.suptitle('Partial dependence of pre diabetes risk factors')
                 
plt.subplots_adjust(right=1,top=.9)
plt.show()
from pdpbox import pdp
pdp_glu_bmi = pdp.pdp_interact(model,diabetes[diabetes_features],features=['Glucose','BMI'], model_features=diabetes_features)
pdp.pdp_interact_plot(pdp_glu_bmi, ['Glucose', 'BMI'],plot_type='grid',x_quantile=True,plot_pdp=True)
pdp.pdp_interact_plot(pdp_glu_bmi, ['Glucose', 'BMI'],plot_type='contour',x_quantile=True,plot_pdp=True)