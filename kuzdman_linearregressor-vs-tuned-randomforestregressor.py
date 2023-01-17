import pandas as pd



import seaborn as sns



from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV



from sklearn.model_selection import cross_val_score
df = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')
df.head()
df = df.drop(columns=['Serial No.']) # because it's absolutely doesn't important
df = df.rename(columns={

    'GRE Score': 'GRE',

    'TOEFL Score': 'TOEFL',

    'University Rating': 'UR',

    'Chance of Admit ': 'Chance',

    'LOR ': 'LOR'

})
df.head()
df.corr()
sns.heatmap(df.corr());
sns.lineplot(x="CGPA", y="Chance", data=df);
sns.lineplot(x="TOEFL", y="GRE", data=df);
sns.lineplot(x="CGPA", y="GRE", data=df);
sns.lineplot(x="SOP", y="LOR", data=df);
X, y = df.drop(columns=['Chance']), df['Chance']
rfr = RandomForestRegressor(random_state=42)

lr = LinearRegression()
params = {

    'n_estimators': range(10, 51, 10),

    'max_depth': range(1, 13, 2),

    'min_samples_leaf': range(1, 8),

    'min_samples_split': range(2, 10, 2)

}
search = GridSearchCV(rfr, params, cv=10, n_jobs=-1)
search.fit(X, y)
search.best_params_
best_rfr = search.best_estimator_
imp = pd.DataFrame(best_rfr.feature_importances_, index=X.columns, columns=['importance'])

imp.sort_values('importance').plot(kind='barh', figsize=(12, 8))
cross_val_score(best_rfr, X, y, cv=10).mean()
cross_val_score(lr, X, y, cv=10).mean()