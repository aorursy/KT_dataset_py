import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import math

import matplotlib



from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier



%matplotlib inline
records = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv', sep=',')

records.head()
print(records['math score'].describe())

print(records.gender.count())
records.select_dtypes('object').nunique()
records.isnull().any()
scores = records.copy()

del scores['gender']

del scores['race/ethnicity']

del scores['parental level of education']

del scores['lunch']

del scores['test preparation course']







def queueHistograms(df):

    for col in df.columns:

        drawHistogram(df[col], col)





def drawHistogram(records, col):

    fig, ax = plt.subplots()

    fig = plt.gcf()

    fig.set_size_inches(10,7)

    plt.hist(records, 10, density=False, color=(0.2, 0.4, 0.6, 0.6))

    plt.xlabel('Scores')

    plt.ylabel('# of students')

    plt.title('Histogram of ' + col)

    plt.grid(True)

    plt.show()

    

queueHistograms(scores)
def graphSpread(name, cols, vals):

    y_pos = np.arange(len(cols))

    fig, ax = plt.subplots()

    fig = plt.gcf()

    fig.set_size_inches(10,7)



    rects1 = ax.bar(cols, vals, align='center', color=(0.2, 0.4, 0.6, 0.6))

    plt.xticks(y_pos, cols)

    ax.set_title(name + ' spread', fontsize=22)

    plt.ylabel('Count')

    autolabel(rects1, ax)

    plt.show()

    return



def getData(df, col):

    cols = df[col].unique()

    counts = df[col].value_counts()

    return cols,counts





def runGraphs(df):

    for col in df.columns:

        c, v = getData(df, col)

        graphSpread(col, c, v)



def autolabel(rects, ax):

    """

    Attach a text label above each bar displaying its height

    """

    for rect in rects:

        height = rect.get_height()

        

        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,

                '%d' % float(height),

        ha='center', va='bottom')





chartDf = records.copy()

del chartDf['math score']

del chartDf['reading score']

del chartDf['writing score']

runGraphs(chartDf)

remove_scores = records[records['math score'] >= 80].copy()

del remove_scores['math score']

del remove_scores['reading score']

del remove_scores['writing score']

total = remove_scores.gender.count()

print(total)
remove_scores.head(10)
def loopData(ds, subject, grade):

    """

    Loop through all of the columns, format the data, and create a graph

    """

    for col in ds.columns:

        df = createData(ds, col, ds.gender.count())

        createPie(df, col, subject, grade)



def createData(ds, header, total):

    """

    Format the data how we want it for the chart

    """

    df = ds.groupby(header, as_index=False).size().reset_index(name='count')

    df['count'] = df['count'].astype(float)

    df['count'] = (df['count'] / total * 100)

    df['count'] = df['count'].round(decimals=0)

    return df



def createPie(data, header, subject, grade):

    """

    Create pie chart.  I am not using this anymore, but it may be helpful for others.

    """

    labels = data[header].unique()

    sizes = data['count'].unique()

    plt.figure(figsize=(15,10))

    fig1, ax1 = plt.subplots()

    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90, radius=1900)

    ax1.axis('equal')

    ax1.set_title(grade + '%+ ' + subject + ' grade by ' + header, bbox={'facecolor':'0.8', 'pad':3}, fontsize=22)

    fig = plt.gcf()

    fig.set_size_inches(10,10)

    plt.show()
def createDfs(df, col):

    """

    Create data frames so that we can create the graphs

    """

    ds = df[df['math score'] >= 80].copy()

    df1 = createData(ds, col, ds.gender.count())



    ds2 = df[df['reading score'] >= 80].copy()

    df2 = createData(ds2, col, ds2.gender.count())



    ds3 = df[df['writing score'] >= 80].copy()

    df3 = createData(ds3, col, ds3.gender.count())



    ds4 = df[(df['writing score'] >= 80) & (df['reading score'] >= 80) & (df['math score'] >= 80)].copy()

    df4 = createData(ds4, col, ds4.gender.count())

    return df1, df2, df3, df4







def runGraphs(df, col):

    """

    Create the graphs to show how the spread was for the top of the classes for each subject.

    """

    

    df1, df2, df3, df4 = createDfs(df, col)

    category_names = list(df1[col])



    results = {

        'Math Score 80%+': list(df1['count']),

        'Writing Score 80%+' : list(df2['count']),

        'Reading Score 80%+' : list(df3['count']),

        'All Scores 80%+' : list(df4['count']),

    }

    

    survey(results, category_names)

    fig = plt.gcf()

    plt.suptitle(col + ' impact on Exam Scores',x=.5)

    fig.set_size_inches(15,7.5) 



    plt.show()







def survey(results, category_names):

    """

    Parameters

    ----------

    results : dict

        A mapping from question labels to a list of answers per category.

        It is assumed all lists contain the same number of entries and that

        it matches the length of *category_names*.

    category_names : list of str

        The category labels.

    """

    labels = list(results.keys())

    data = np.array(list(results.values()))

    data_cum = data.cumsum(axis=1)

    category_colors = plt.get_cmap('twilight')(

        np.linspace(0.15, 0.85, data.shape[1]))



    fig, ax = plt.subplots(figsize=(9.2, 5))

    ax.invert_yaxis()

    ax.xaxis.set_visible(False)

    ax.set_xlim(0, np.sum(data, axis=1).max())



    for i, (colname, color) in enumerate(zip(category_names, category_colors)):

        widths = data[:, i]

        starts = data_cum[:, i] - widths

        ax.barh(labels, widths, left=starts, height=0.5,

                label=colname, color=color)

        xcenters = starts + widths / 2



        r, g, b, _ = color

        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'

        for y, (x, c) in enumerate(zip(xcenters, widths)):

            ax.text(x, y, str(int(c)), ha='center', va='center',

                    color=text_color)

    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),

              loc='lower left', fontsize='small')



    return fig, ax
# Create all of the graphs

for col in remove_scores.columns:

    runGraphs(records, col)
# Get a count of everything

records['math high mark'] = records['math score'] >= 80

records['reading high mark'] = records['reading score'] >= 80

records['writing high mark'] = records['writing score'] >= 80

records['all high mark'] = (records['math score'] >= 80) &(records['reading score'] >= 80) &(records['writing score'] >= 80)



print(records['math high mark'].value_counts())

print(records['reading high mark'].value_counts())

print(records['writing high mark'].value_counts())

print(records['all high mark'].value_counts())
df = records.groupby('gender', as_index=False).size().reset_index(name='count')
print(df)
def createDBData(df, label):

    """

    Create a new data frame for the bars to say how many got high marks, 

    and how many did not.

    """

    newDF = pd.DataFrame(data={label: [], 'high mark': [], 'not high mark': []})

    for col in df[label].unique():

        tds = records[df[label] == col]

        total = len(tds)

        tdf = tds.groupby('math high mark', as_index=False).size().reset_index(name='count')

        high = tdf.iloc[1]['count']

        low = tdf.iloc[0]['count']



        newDF = newDF.append({label: col, 'high mark': high, 'not high mark': low}, ignore_index=True)

    return newDF
def autolabel2(rects, ax):

    """Attach a text label above each bar in *rects*, displaying its height."""

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 1),  # 3 points vertical offset

                    textcoords="offset points",

                    ha='center', va='bottom', color='black')





def drawBars(df, col):

    labels = df[col].unique()

    high_mark = df['high mark']

    low_mark = df['not high mark']



    x = np.arange(len(labels))  # the label locations

    width = 0.35  # the width of the bars



    fig, ax = plt.subplots()

    rects1 = ax.bar(x - width/2, high_mark, width, label='Score >= 80', align='center', color=(0.2, 0.4, 0.6, 0.6))

    rects2 = ax.bar(x + width/2, low_mark, width, label='Score < 80', align='center', color = (0.6, 0.4, 0.2, 0.6))



    # Add some text for labels, title and custom x-axis tick labels, etc.

    ax.set_ylabel('Count')

    ax.set_title('Math Count by score and ' + col)

    ax.set_xticks(x)

    ax.set_xticklabels(labels)

    ax.legend()

    autolabel2(rects1, ax)

    autolabel2(rects2, ax)



    fig.tight_layout()



    fig.set_size_inches(15,10) # or (4,4) or (5,5) or whatever



    plt.show()
for col in remove_scores.columns:

    newDF = createDBData(records, col)

    drawBars(newDF, col)
def convertToGrade(percent):

    """

    Convert the % grade to a letter grade.

    """

    if(percent >= 90):

        return 'A'

    if(percent >=80):

        return 'B'

    if(percent >=70):

        return 'C'

    if(percent >=60):

        return 'D'

    return 'F'
records['math grade'] = records.apply(lambda x: convertToGrade(x['math score']), axis = 1)

records['reading grade'] = records.apply(lambda x: convertToGrade(x['reading score']), axis = 1)

records['writing grade'] = records.apply(lambda x: convertToGrade(x['writing score']), axis = 1)





print(records['math grade'].value_counts())

print(records['reading grade'].value_counts())

print(records['writing grade'].value_counts())

histDF = records.copy()

histDF.head()
# Delete everything but the grades for charting

del histDF['gender']

del histDF['race/ethnicity']

del histDF['parental level of education']

del histDF['test preparation course']

del histDF['math score']

del histDF['reading score']

del histDF['writing score']

del histDF['math high mark']

del histDF['reading high mark']

del histDF['writing high mark']

del histDF['all high mark']

del histDF['lunch']



histDF.head()
histDFSorted = histDF.sort_values(by=['math grade'], ascending=False)

histDFSorted.head()

queueHistograms(histDF)
histDFSorted['math grade'].value_counts(sort=False)
records.isnull().any()
modelDF = records.copy()
def passOrFail(percent):

    """

    Create a pass/fail column

    """

    if(percent >= 60):

        return 1

    return 0
modelDF['average score'] = (modelDF['math score'] + modelDF['reading score'] + modelDF['writing score'])  / 3

modelDF['average score'] = modelDF['average score'].round(decimals=0)

modelDF['average grade'] = modelDF.apply(lambda x : convertToGrade(x['average score']), axis=1)

records['average grade'] = modelDF['average grade']



modelDF['passed'] = modelDF.apply(lambda x: passOrFail(x['average score']), axis = 1)

modelDF.head()
def convertColumn(df, col):

    """

    We can't use Strings on some of these columns, so let's do a simple conversion to numeric

    """

    i = 0

    newDF = df.copy()

    for val in df[col].unique():

        newDF[col] = newDF[col].replace(val, i)

        i = i + 1

    return newDF[col]





modelDF['gender'] = convertColumn(modelDF,'gender')

modelDF['race/ethnicity'] = convertColumn(modelDF,'race/ethnicity')

modelDF['parental level of education'] = convertColumn(modelDF,'parental level of education')

modelDF['lunch'] = convertColumn(modelDF,'lunch')

modelDF['test preparation course'] = convertColumn(modelDF,'test preparation course')







modelDF.head()



y = modelDF[['average grade']].copy()

y.head()
#features = ['gender', 'race/ethnicity', 'lunch', 'test preparation course', 'math score','reading score', 'writing score', 'math high mark', 'reading high mark', 'writing high mark', 'all high mark', 'math grade', 'reading grade', 'writing grade', 'average score']

#features = ['gender', 'race/ethnicity', 'lunch', 'test preparation course', 'math score','reading score', 'writing score', 'math high mark', 'reading high mark', 'writing high mark', 'all high mark']

#features = ['gender', 'race/ethnicity', 'lunch', 'test preparation course', 'math score','reading score', 'writing score']

#features = ['gender', 'race/ethnicity', 'lunch', 'test preparation course', 'math score','reading score', 'writing score', 'math high mark', 'reading high mark', 'writing high mark', 'all high mark', 'average score']





features = ['gender', 'race/ethnicity', 'lunch', 'test preparation course']





X = modelDF[features].copy()
X.columns
y.columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12)
math_classifier = DecisionTreeClassifier(max_leaf_nodes=35, random_state=337)

math_classifier.fit(X_train, y_train)
predictions = math_classifier.predict(X_test)
accuracy_score(y_true = y_test, y_pred = predictions)
y2 = modelDF['passed']



X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.25, random_state=12)



math_classifier = DecisionTreeClassifier(max_leaf_nodes=3, random_state=337)

math_classifier.fit(X_train, y_train)



predictions = math_classifier.predict(X_test)

accuracy_score(y_true = y_test, y_pred = predictions)
features = ['gender', 'race/ethnicity', 'lunch', 'test preparation course', 'reading score', 'writing score']





y3 = modelDF['math grade']

X = modelDF[features].copy()





X_train, X_test, y_train, y_test = train_test_split(X, y3, test_size=0.25, random_state=12)



math_classifier = DecisionTreeClassifier(max_leaf_nodes=40, random_state=337)

math_classifier.fit(X_train, y_train)



predictions = math_classifier.predict(X_test)

accuracy_score(y_true = y_test, y_pred = predictions)
y4 = modelDF.apply(lambda x: passOrFail(x['math score']), axis = 1)



features = ['gender', 'race/ethnicity', 'lunch', 'test preparation course', 'reading score', 'writing score']

X = modelDF[features].copy()





X_train, X_test, y_train, y_test = train_test_split(X, y4, test_size=0.25, random_state=12)



math_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=337)

math_classifier.fit(X_train, y_train)



predictions = math_classifier.predict(X_test)

accuracy_score(y_true = y_test, y_pred = predictions)

secondDF = records.copy()

secondDF.head()
secondDF = pd.get_dummies(secondDF)
secondDF.head()
secondDF.columns
using = secondDF.copy()
features = ['gender_female', 'gender_male', 'race/ethnicity_group A',

       'race/ethnicity_group B', 'race/ethnicity_group C',

       'race/ethnicity_group D', 'race/ethnicity_group E',

       "parental level of education_associate's degree",

       "parental level of education_bachelor's degree",

       'parental level of education_high school',

       "parental level of education_master's degree",

       'parental level of education_some college',

       'parental level of education_some high school', 'lunch_free/reduced',

       'lunch_standard', 'test preparation course_completed',

       'test preparation course_none']



X = using[features]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12)

math_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=337)

math_classifier.fit(X_train, y_train)



predictions = math_classifier.predict(X_test)

accuracy_score(y_true = y_test, y_pred = predictions)
X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.25, random_state=12)

math_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=337)

math_classifier.fit(X_train, y_train)



predictions = math_classifier.predict(X_test)

accuracy_score(y_true = y_test, y_pred = predictions)
features = ['gender_female', 'gender_male', 'race/ethnicity_group A',

       'race/ethnicity_group B', 'race/ethnicity_group C',

       'race/ethnicity_group D', 'race/ethnicity_group E',

       "parental level of education_associate's degree",

       "parental level of education_bachelor's degree",

       'parental level of education_high school',

       "parental level of education_master's degree",

       'parental level of education_some college',

       'parental level of education_some high school', 'lunch_free/reduced',

       'lunch_standard', 'test preparation course_completed',

       'test preparation course_none', 'reading grade_A',

       'reading grade_B', 'reading grade_C', 'reading grade_D',

       'reading grade_F', 'writing grade_A', 'writing grade_B',

       'writing grade_C', 'writing grade_D', 'writing grade_F','math grade_A', 'math grade_B',

       'math grade_C', 'math grade_D', 'math grade_F' ]



X = using[features]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12)

math_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=337)

math_classifier.fit(X_train, y_train)



predictions = math_classifier.predict(X_test)

accuracy_score(y_true = y_test, y_pred = predictions)
features = ['gender', 'race/ethnicity', 'lunch', 'test preparation course', 'math score','reading score', 'writing score', 'math high mark', 'reading high mark', 'writing high mark', 'all high mark']



X = modelDF[features].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12)

math_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=337)

math_classifier.fit(X_train, y_train)



predictions = math_classifier.predict(X_test)

accuracy_score(y_true = y_test, y_pred = predictions)

features = ['gender_female', 'gender_male', 'race/ethnicity_group A',

       'race/ethnicity_group B', 'race/ethnicity_group C',

       'race/ethnicity_group D', 'race/ethnicity_group E',

       "parental level of education_associate's degree",

       "parental level of education_bachelor's degree",

       'parental level of education_high school',

       "parental level of education_master's degree",

       'parental level of education_some college',

       'parental level of education_some high school', 'lunch_free/reduced',

       'lunch_standard', 'test preparation course_completed',

       'test preparation course_none', 'reading grade_A',

       'reading grade_B', 'reading grade_C', 'reading grade_D',

       'reading grade_F', 'writing grade_A', 'writing grade_B',

       'writing grade_C', 'writing grade_D', 'writing grade_F','math grade_A', 'math grade_B',

       'math grade_C', 'math grade_D', 'math grade_F' ]



X = using[features]



X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.25, random_state=12)

math_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=337)

math_classifier.fit(X_train, y_train)



predictions = math_classifier.predict(X_test)

accuracy_score(y_true = y_test, y_pred = predictions)
features = ['gender', 'race/ethnicity', 'lunch', 'test preparation course', 'math score','reading score', 'writing score', 'math high mark', 'reading high mark', 'writing high mark', 'all high mark']



X = modelDF[features].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.25, random_state=12)

math_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=337)

math_classifier.fit(X_train, y_train)



predictions = math_classifier.predict(X_test)

accuracy_score(y_true = y_test, y_pred = predictions)
