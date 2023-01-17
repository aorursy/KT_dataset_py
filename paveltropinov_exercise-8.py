# Load all necessary libs which will be used further
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

print('Setup completed.')
# Load netflix titles' data
netflix_data = pd.read_csv('../input/my-dataset/netflix_titles.csv', index_col='show_id')

# Draw plot describes counts of content types
types_count = netflix_data.groupby('type').type.count()
plt.title('Counts of different content types')
sns.barplot(x=types_count.index, y=types_count)
plt.xlabel('Content type')
plt.ylabel('Total count')

# Group by directors, calculate count of films they have been directed and store results to output CSV file
netflix_data.groupby('director').director.count().to_csv('output.csv', index=True, header=False)
# Load houses data and obtain column for predictions
X = pd.read_csv('../input/my-dataset/Nedvijimost.csv', index_col='ID объекта')
print(houses.describe())
y = X['Стоимость (т.руб.)']

# Obtain numerical and categorical columns to make appropriate imputions
numerical_columns = [column for column in houses.columns if houses[column].dtype in ['int64', 'float64']]
categorical_columns = [column for column in houses.columns if houses[column].nunique() < 10 and houses[column].dtype == "object"]
# Prepare pipelines for easy use several times
numerical_transformer = SimpleImputer(strategy='constant')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_columns),
    ('cat', categorical_transformer, categorical_columns)])

# Try different counts of estimators to find the best one by the lowest Mean Absolute Error
print()
scores = {}
for i in range(1, 21):
    current_estimators = i * 50
    scores[current_estimators] = get_mae_score(preprocessor, current_estimators, X, y)
    print('Calculated MAE for', current_estimators, 'estimators =', scores[current_estimators])

# Find count of estimators which produces the lowest MAE
estimators = min(scores, key=scores.get)
print('The lowest MAE =', scores[estimators], 'found for', estimators, 'estimators count. It will be used to create the model.')

# Assing best_model due to obtained estimators count
best_model = RandomForestRegressor(estimators, random_state=0)


# Define function to calculate MAE for specified count of estimators
def get_mae_score(preprocessor, estimators, X, y):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(estimators, random_state=0))]) 
    return (cross_val_score(pipeline, X, y, cv=3, scoring='neg_mean_absolute_error') * -1).mean()