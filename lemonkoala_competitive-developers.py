import numpy   as np
import pandas  as pd
import seaborn as sns

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot  as plt

matplotlib.rcParams['figure.figsize'] = (20, 7)
survey = pd.read_csv('../input/survey_results_public.csv', dtype='category')
statements = {
    'AgreeDisagree1': "I feel a sense of kinship or connection to other developers",
    'AgreeDisagree2': "I think of myself as competing with my peers",
    'AgreeDisagree3': "I'm not as good at programming as most of my peers"
}

general_counts = []

for i in range(3):
    column = f"AgreeDisagree{i + 1}"

    survey[column] = survey[column].cat.reorder_categories([
        'Strongly agree',
        'Agree',
        'Neither Agree nor Disagree',
        'Disagree',
        'Strongly disagree',
    ])

    df = survey[column].value_counts().sort_index().reset_index(name='count')
    df.rename(columns={ 'index': 'answer' }, inplace=True)
    df['statement'] = statements[column]

    general_counts.append(df)

ax = plt.subplot()
ax.set_title("To what extent do you agree or disagree with each of the following statements?")

sns.barplot(
    data=pd.concat(general_counts),
    x='count',
    y='answer',
    hue='statement',
    ax=ax
)

plt.savefig('statements.png')
plt.show()
def percent_plot(
    df,
    bar=0.85,
    orientation='h',
    title='',
    palette=sns.color_palette("coolwarm", 5),
    figsize=None
):
    # get percentages from counts
    df = df.div(df.sum(axis=1), axis=0)

    ticks = range(len(df))
    last_values = [0 for tick in ticks]

    # if graphing horizontally, we want to order
    # from top to bottom
    if orientation == 'h':
        df = df.iloc[::-1]
        
    if figsize is not None:
        plt.figure(figsize=figsize)

    for i, col in enumerate(df.columns):
        if orientation == 'h':
            plt.xlabel("Percent")
            plt.yticks(ticks, df.index.tolist())
            plt.barh(
                ticks,
                df[col].tolist(),
                height=bar,
                left=last_values,
                color=palette[i]
            )

        else:
            plt.ylabel("Percent")
            plt.xticks(ticks, df.index.tolist())
            plt.bar(
                ticks,
                df[col].tolist(),
                width=bar,
                bottom=last_values,
                color=palette[i]
            )

        last_values = [
            last + curr
            for last, curr in zip(last_values, df[col].tolist())
        ]

    plt.legend(
        title='Competitive',
        handles=[
            mpatches.Patch(color=palette[i], label=col)
            for i, col in enumerate(df.columns)
        ]
    )

    plt.title(title)
    plt.show()
def calculate_column_counts(column):
    df = survey.groupby([column, 'AgreeDisagree2']).size().reset_index(name='Count')
    df = df.pivot(column, 'AgreeDisagree2', 'Count')
    return df

# for columns with multiple responses (so slow though)
def calculate_multivalue_column_counts(column):
    s = survey[column].astype(str).str.split(';').apply(pd.Series, 1).stack()
    s.index = s.index.droplevel(-1)
    s.name  = column

    return survey[['AgreeDisagree2']].           \
            join(s).                             \
            groupby([column, 'AgreeDisagree2']). \
            size().                              \
            reset_index(name='Count').           \
            pivot(column, 'AgreeDisagree2', 'Count')
def graph_column(column, calculate=calculate_column_counts, **kwargs):
    df = calculate(column)
    percent_plot(df, title=f"Competitiveness by {column}", **kwargs)
survey['Age'].cat.reorder_categories([
    'Under 18 years old',
    '18 - 24 years old',
    '25 - 34 years old',
    '35 - 44 years old',
    '45 - 54 years old',
    '55 - 64 years old',
    '65 years or older'
], inplace=True)

graph_column('Age')
graph_column('Country', figsize=(20, 40))
graph_column('Gender', calculate=calculate_multivalue_column_counts)
graph_column('SexualOrientation', calculate=calculate_multivalue_column_counts)
graph_column('NumberMonitors')
survey['WakeTime'].cat.reorder_categories([
    'Before 5:00 AM',
    'Between 5:00 - 6:00 AM',
    'Between 6:01 - 7:00 AM',
    'Between 7:01 - 8:00 AM',
    'Between 8:01 - 9:00 AM',
    'Between 9:01 - 10:00 AM',
    'Between 10:01 - 11:00 AM',
    'Between 11:01 AM - 12:00 PM',
    'After 12:01 PM',
    'I work night shifts',
    'I do not have a set schedule'
], inplace=True)
graph_column('WakeTime')
survey['HoursComputer'].cat.reorder_categories([
    'Less than 1 hour',
    '1 - 4 hours',
    '5 - 8 hours',
    '9 - 12 hours',
    'Over 12 hours'
], inplace=True)
graph_column('HoursComputer')
survey['HoursOutside'].cat.reorder_categories([
    'Less than 30 minutes',
    '30 - 59 minutes',
    '1 - 2 hours',
    '3 - 4 hours',
    'Over 4 hours'
], inplace=True)
graph_column('HoursOutside')
graph_column('HackathonReasons', calculate=calculate_multivalue_column_counts)
graph_column('Exercise')
survey['SkipMeals'].cat.reorder_categories([
    'Never',
    '1 - 2 times per week',
    '3 - 4 times per week',
    'Daily or almost every day',
], inplace=True)
graph_column('SkipMeals')
survey['JobSatisfaction'].cat.reorder_categories([
    'Extremely satisfied',
    'Moderately satisfied',
    'Slightly satisfied',
    'Neither satisfied nor dissatisfied',
    'Slightly dissatisfied',
    'Moderately dissatisfied',
    'Extremely dissatisfied',
], inplace=True)

graph_column('JobSatisfaction')
survey['CareerSatisfaction'].cat.reorder_categories([
    'Extremely satisfied',
    'Moderately satisfied',
    'Slightly satisfied',
    'Neither satisfied nor dissatisfied',
    'Slightly dissatisfied',
    'Moderately dissatisfied',
    'Extremely dissatisfied',
], inplace=True)

graph_column('CareerSatisfaction')
graph_column('HopeFiveYears')
survey['JobSearchStatus'].cat.reorder_categories([
    'I am actively looking for a job',
    'I’m not actively looking, but I am open to new opportunities',
    'I am not interested in new job opportunities',
], inplace=True)

graph_column('JobSearchStatus')
def plot_rank(name, num):
    df = get_rank_df(name, num)
    sns.boxplot(
        data=df,
        x='Rank',
        y=name,
        hue='AgreeDisagree2'
    );
    
def get_rank_df(name, num):
    columns = [
        f"{name}{i + 1}"
        for i in range(num)
    ]

    for column in columns:
        survey[column] = survey[column].astype(float)
    
    columns.append('AgreeDisagree2')

    df = pd.melt(
        survey[columns],
        id_vars=['AgreeDisagree2'],
        value_vars=columns[:num],
        var_name=name,
        value_name='Rank'
    ).dropna()

    return df
graph_column('EthicsChoice')
graph_column('EthicsReport')
graph_column('EthicsResponsible')
graph_column('EthicalImplications')
graph_column('AIResponsible')
survey['ConvertedSalary'] = survey['ConvertedSalary'].astype('float')

sns.boxplot(
    data=survey[['AgreeDisagree2', 'ConvertedSalary']].dropna(),
    x='AgreeDisagree2',
    y='ConvertedSalary',
    palette=sns.color_palette("coolwarm", 5),
    showfliers=False
)

plt.title('ConvertedSalary by Competitiveness')
plt.show()
survey['SurveyTooLong'].cat.reorder_categories([
    'The survey was too short',
    'The survey was an appropriate length',
    'The survey was too long'
], inplace=True)

graph_column('SurveyTooLong')
survey['SurveyEasy'].cat.reorder_categories([
    'Very easy',
    'Somewhat easy',
    'Neither easy nor difficult',
    'Somewhat difficult',
    'Very difficult',
], inplace=True)

graph_column('SurveyEasy')
null_counts = pd.DataFrame({
    'AgreeDisagree2': survey['AgreeDisagree2'],
    'NullCount':      survey.isnull().sum(axis=1)
})

sns.boxplot(
    data=null_counts,
    x='AgreeDisagree2',
    y='NullCount',
    palette=sns.color_palette("coolwarm", 5),
    showfliers=False
)

plt.title('Number of null entries by competitiveness')
plt.show()
survey_clf = survey[[
    'AgreeDisagree2',
    'Age',
    'SurveyEasy',
    'NumberMonitors',
    'SkipMeals',
    'HoursOutside',
    'EthicalImplications'
]].dropna()

def get_encoder(column):
    ordered_categories = survey[column].cat.categories.tolist()
    return lambda x: ordered_categories.index(x)

for column in survey_clf.select_dtypes(include=['category']):
    survey_clf[column] = survey_clf[column].apply(get_encoder(column))

survey_clf.head()
from sklearn import tree
from sklearn.model_selection import train_test_split

X = survey_clf.drop('AgreeDisagree2', axis=1)
y = survey_clf['AgreeDisagree2']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.33,
    random_state=42
)

clf = tree.DecisionTreeClassifier(random_state=42)
clf = clf.fit(X_train, y_train)
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cnf_matrix = confusion_matrix(y_test, clf.predict(X_test))
plot_confusion_matrix(cnf_matrix, classes=survey['AgreeDisagree2'].cat.categories.tolist(), title='Confusion matrix')

plt.show()
import graphviz

dot_data = tree.export_graphviz(
    clf,
    out_file=None,
    max_depth=2,
    feature_names=X.columns.tolist(),
    class_names=survey['AgreeDisagree2'].cat.categories.tolist(),
    filled=True,
    rounded=True
)
graph = graphviz.Source(dot_data)
graph
df = survey.groupby(['AgreeDisagree1', 'AgreeDisagree2']). \
    size().                                                \
    reset_index(name="Count").                             \
    pivot("AgreeDisagree1", "AgreeDisagree2", "Count")

ax = sns.heatmap(df, annot=True, fmt="d")
ax.set(
    title="Competitiveness and Kinship",
    xlabel="I think of myself as competing with my peers",
    ylabel="I feel a sense of kinship or connection to other developers"
)
plt.show()
