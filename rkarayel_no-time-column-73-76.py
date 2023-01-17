%matplotlib inline
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.rcParams["axes.labelsize"] = 15
df = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
df.head(2)
df.shape
df.info()
df.describe().T
nan_cols = df.isna().sum()
nan_cols[nan_cols>0]
df.columns = [col.lower().strip() for col in df.columns]
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
# Set up FacetGrid
g = sns.FacetGrid(df,
                  row="sex",
                  hue="sex",
                  aspect=10,
                  height=5,
                  palette=['#4285F4', '#FBBC05'],     
                 )
# Map Plot and Plot Shape (outline)
g.map(sns.kdeplot, "age", clip_on=False, shade=True, alpha=1, lw=1.5, bw=1.2)
g.map(sns.kdeplot, "age", clip_on=False, color="w", lw=3, bw=1.2)

# X-Axis properties
g.map(plt.axhline, y=0, lw=6)
g.map(plt.xticks, fontsize=40, color='#2F2F2F')
g.map(plt.axhline, y=0, lw=2, clip_on=False)

g.map(plt.axhline, y=0, lw=2, clip_on=False)
g.fig.subplots_adjust(hspace= -.20)


# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, 'Sex = '+label, fontweight="bold", color=color, size=50,
            ha="left", va="center", transform=ax.transAxes)

g.map(label, "age")

# Set the subplots to overlap
g.fig.subplots_adjust(hspace=-.25)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[])
g.despine(bottom=True, left=True)

plt.show()
g1 = sns.FacetGrid(df,
                  aspect=3,
                  height=5,
                 )
g1.map(sns.distplot, 'age', kde=False, hist_kws={"alpha": 1, "color": "#31AF94", 'edgecolor': "w"})
g1.map(sns.distplot, 'age', kde=False, hist_kws={"alpha": 1, "edgecolor": "#247D96", "histtype": "step", "linewidth": 3})


g1.despine(bottom=True, left=True)
g1.set(yticks=[])

plt.show()
sns.set(style="whitegrid", rc={"axes.facecolor": (0, 0, 0, 0)}, context='talk')

g = sns.FacetGrid(df,
                  col='high_blood_pressure',
                  aspect=2,
                  height=5,
                  margin_titles=True,
                 )

g.map(plt.axhline, y=0, lw=8, color='darkgrey')
g.map(sns.countplot, 'smoking', order=[0, 1], palette='GnBu', edgecolor='#57534F')
g.despine(bottom=False, left=True)

plt.show()
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}, context='talk')

g = sns.FacetGrid(df,
                  row='death_event',
                  hue='sex',
                  aspect=6,
                  height=3,
                  palette=['#34A853', '#EA4335'],     
                 )

g.map(sns.kdeplot, 'age', shade=True, alpha=0.5, lw=1.5, bw=1.2)
g.map(sns.kdeplot, 'age', lw=3, bw=1.2)

g.despine(bottom=False, left=True)
g.set(yticks=[])
g.add_legend()


plt.show()
sns.set_context("talk")
f = plt.figure(figsize=(18, 8))
gs = f.add_gridspec(2, 2)

with sns.axes_style("dark", {"axes.facecolor": "#FDE9F5"}):
    ax = f.add_subplot(gs[0, 0])
    sns.boxplot('age', 'diabetes', data=df, orient='h', palette={0:'#34A853', 1:'#EA4335'})

with sns.axes_style("dark", {"axes.facecolor": "#EFE2F4"}):
    ax = f.add_subplot(gs[0, 1])
    sns.boxplot('age', 'smoking', data=df, orient='h', palette={0:'#34A853', 1:'#EA4335'})

with sns.axes_style("dark", rc={"axes.facecolor": "#E1DAF4"}):
    ax = f.add_subplot(gs[1, 0])
    sns.boxplot('age', 'high_blood_pressure', data=df, orient='h', palette={0:'#34A853', 1:'#EA4335'})

with sns.axes_style("dark", rc={"axes.facecolor": "#D2D3F3"}):
    ax = f.add_subplot(gs[1, 1])
    sns.boxplot('age', 'anaemia', data=df, orient='h', palette={0:'#34A853', 1:'#EA4335'})

sns.despine(left=True)

f.tight_layout()
sns.set_context(context='notebook')
j = sns.jointplot(x='ejection_fraction', y='platelets', ratio=4, height=7, data=df, kind='kde',color='#093E58')
j.plot_joint(plt.scatter, c="c", s=40, linewidth=1, marker="+")
plt.show()
sns.jointplot(x='serum_sodium', y='serum_creatinine', ratio=4, height=7,data=df, kind='hex',color='#C11C55')
plt.tight_layout()
plt.show()
fig, axs = plt.subplots(1, 2, figsize=(20,6))
sns.countplot(y='death_event', data=df, ax=axs[0], palette={0: '#34A853', 1:'#EA4335'})
sns.scatterplot(x='age', y='time', data=df, hue=df['death_event'].to_list(), palette={0: '#34A853', 1:'#EA4335'}, ax=axs[1], legend=False)
sns.regplot(x='age', y='time',data=df[df['death_event']==0], ax=axs[1], color='#34A853', scatter_kws={"s": 0})
sns.regplot(x='age', y='time',data=df[df['death_event']==1], ax=axs[1], color='#EA4335',scatter_kws={"s": 0})
sns.despine(left=False)
plt.show()
plt.figure(figsize=(5,8))
heatmap = sns.heatmap(df.corr()[['death_event']].sort_values(by='death_event', ascending=False), annot=True, cmap='RdYlGn', cbar=False)
heatmap.set_title('Features Correlating with Death Event', fontdict={'fontsize':14}, pad=10)
plt.show()
X = df.drop(['death_event', 'time'], axis=1)
y = df['death_event']
print(X.shape)
print(y.shape)
from scipy import stats

def make_skew_df(df, numerical_cols):
    init_skews = []
    new_skews = []
    
    for col in numerical_cols:
        current_skew_val = df[col].skew()
        init_skews.append(current_skew_val)
        
        yeo, yeo_lmbda = stats.yeojohnson(df[col])
        new_skews.append(pd.Series(yeo).skew())
    
    skew_diffs = (np.abs(init_skews).round(2)-np.abs(new_skews).round(2))
    
    
    df_skew = pd.DataFrame({
        'Feature': numerical_cols, 
        'Skew_Before_Transformation': init_skews, 
        'Skew_After_Transformation': new_skews, 
        'Skew_Diff': skew_diffs
    }).sort_values(by='Skew_Diff', ascending=False).reset_index(drop=True)
    return df_skew
numerical_features = X.dtypes[X.dtypes != 'object'].index
df_skews = make_skew_df(X, numerical_features)
df_skews
g = sns.FacetGrid(df_skews,
                  aspect=3,
                  height=5,
                 )
g.map(sns.distplot, 'Skew_Before_Transformation', kde=False, hist_kws={"alpha": 0.8, "color": "#CEC7C4"})
g.map(sns.distplot, 'Skew_After_Transformation', kde=False, hist_kws={"alpha": 1, "color": "#34A853", 'edgecolor': "#FF751B", "linewidth": 2})



g.despine(bottom=True, left=True)
g.set(yticks=[])
g.axes[0,0].set_xlabel('Skew Transformation')
plt.show()
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
non_binary_features = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_sodium']
num_yeo_transformer=Pipeline(steps=[
    ('Scale', StandardScaler()),
    ('Skew', PowerTransformer(method='yeo-johnson', standardize=False))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', num_yeo_transformer, non_binary_features)
], remainder='passthrough')
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
models=[
    ('XGBClassifier', XGBClassifier(random_state=42)),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42)),
    ('KNeighbors', KNeighborsClassifier()),
    ('SVM', SVC(random_state=42)),
    ('LogReg', LogisticRegression(random_state=42))
]
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imb_Pipeline
def eval_model(model, splits, reps):
    cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=reps, random_state=42)
    acc_score = np.mean(cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1))
    return acc_score
for model_name, model in models:
    pipe = imb_Pipeline(steps=[
        ('prep', preprocessor),
        ('sampling', SMOTE(sampling_strategy='minority', random_state=42)),
        ('model', model)
    ])
    result = eval_model(pipe, 10, 5)
    print(f'{model_name}: {round(result, 3)}')
from sklearn.model_selection import GridSearchCV, train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf = SVC(random_state=42)
params = { 
    'model__C': [0.1, 1, 2, 10, 100],
    'model__kernel': ['rbf'],
    'model__gamma': [0.01, 0.05, 1, 10]
}
pipe = imb_Pipeline(steps=[
        ('prep', preprocessor),
        ('sampling', SMOTE(sampling_strategy='minority', random_state=7)),
        ('model', clf)
    ])
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
gridsearch = GridSearchCV(pipe, params, cv=cv, n_jobs=-1)
gridsearch.fit(X_train, y_train)
print(f'Best Paramters:\n\t-> {gridsearch.best_params_}')
gridsearch.best_score_
best_estimator = gridsearch.best_estimator_
y_pred = best_estimator.predict(X_valid)
from sklearn.metrics import confusion_matrix, classification_report
cnf_matrix = confusion_matrix(y_valid, y_pred)
plt.figure(figsize=(8,4))
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="binary" ,fmt='g', cbar=False)

plt.title('Confusion Matrix', fontsize=16)
plt.ylabel('Actual label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.show()
y_valid.value_counts().to_frame()
print(classification_report(y_valid, y_pred))