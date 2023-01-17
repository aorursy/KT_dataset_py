#Importing the usual packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
%matplotlib inline

#Make graphs pretty
sns.set()
sns.set_palette(sns.cubehelix_palette(10, start=1.7, reverse=True))

#Below are functions which I wrote to plot specific types of questions in this survey. The first plots questions like
#which tools do you use at work, select all that apply." The second function plots several questions that are all related,
#such as "How important are the following factors in your job search?" where the answers for each factor are "Not important,
#somewhat important, very important."

#Writing functions like this isn't really necessary, but I found myself writing similar code multple times, which is a good
#clue that you can generalize the code and resuse it multple times, which is basically all a function is.

def select_all_that_apply_plot(df, question, figsize=(12,36)):
    """Takes a dataframe and multiple answer question stem, returns barplot of counts in descending order
    :param df: a DataFrame containing survey results
    :param question_stem: a string containing the question name
    :param figsize: a tuple containing desired figure dimenstions, default = (12,36)
    """
    filtered = pd.DataFrame(df.loc[:,df.columns.str.startswith(question)])
    split = filtered[question].dropna().str.split(',').tolist()
    
    #There has to be a beter way to do this. Nested for loops usually aren't the answer.
    flattened = []
    for i in split:
        for j in i:
            flattened.append(j)
            
    flattened_DF = pd.DataFrame(flattened, columns=[question])
    plt.figure(figsize=(12,6))

    ax = sns.countplot(y=question, data=flattened_DF, order=flattened_DF[question].value_counts().index);
    plt.setp(ax.get_yticklabels(), fontsize=15)
    plt.ylabel('');
    plt.title(question + ', N = ' + str(len(filtered)))
    plt.show()
    
    return

#I ended up not using this function, but left it commented out here just in case. I found that graphing these questions using
#numerical substitutions offered more insight.
'''def multi_plot_from_category(df, question_stem, figsize=(12,36)):
    """Takes a dataframe and a question stem to a multiple-part question, returns barplots of counts
    :param df: a DataFrame containing survey results
    :param question_stem: a string, the leading text of the question
    :param figsize: a tuple containing desired figure dimenstions, default = (12,36)
    """
    #create a new DataFrame made of only the columns we care about
    filtered = pd.DataFrame(df.loc[:,df.columns.str.startswith(question_stem)])
    
    #create a new DataFrame with the counts from the columns of the filtered DataFrame
    counts = pd.DataFrame([filtered[x].value_counts(dropna=True) for x in filtered])
    
    num_columns = len(counts)
    plt.figure(figsize=figsize)
    
    #Create subplots for each question
    for i in range(num_columns):
        plt.subplot(math.ceil(num_columns/4),4,i+1)
        plt.title(counts.index[i][len(question_stem):])
        sns.barplot(x=counts.columns, y=counts.iloc[i])
        plt.ylabel('Count')

    plt.show()
    return'''

def multi_plot_hist(df, question_stem, figsize=(24,18)):
    """Takes a dataframe and a question stem to a multiple-part question, returns histogram of counts. Useful for
    percentage responses.
    :param df: a DataFrame containing survey results
    :param question_stem: a string, the leading text of the question
    :param figsize: a tuple containing desired figure dimenstions, default = (12,36)
    """
    #create a new DataFrame made of only the columns we care about
    filtered = pd.DataFrame(df.loc[:,df.columns.str.startswith(question_stem)]).dropna()
    
    num_columns = len(filtered.columns)
    plt.figure(figsize=figsize)
    
    #Create subplots for each question
    for i in range(num_columns):
        plt.subplot(math.ceil(num_columns/3),3,i+1)
        plt.title(filtered.columns[i][len(question_stem):])
        plt.xlabel('Percentage')
        plt.hist(filtered[filtered.columns[i]], rwidth=0.8)

    plt.show()
    return filtered

def replace_usefulness(df, question_stem):
    """Takes a DataFrame and a question stem. Replaces 'Very Useful' with 1, 'Somewhat Useful with 0.5. and 'Not Useful' with 0
    :param df: DataFrame
    :question_stem: a string containing the quesiton stem"""
    
    #Create a new DataFrame from only the questions we care about
    filtered = pd.DataFrame(df.loc[:,df.columns.str.startswith(question_stem)])
    
    #Replace words with useful numbers
    filtered.replace({
        'Very useful' : 1,
        'Somewhat useful' : 0.5,
        'Not Useful' : 0,
        np.nan : 0
    }, inplace=True)
    
    return filtered

def plot_usefulness_questions(df, question_stem, figsize=(12,36), drop_last=None):
    """Plots a scaled frequency chart for multiple, related questions about usefuless.
    :param df: dataframe
    :param question_stem: a string containing the question stem
    :param figsize: tuple for figure size, default = (12,36)
    :param drop_last: number of questions to drop from the end"""
    
    #Use our replace_usefuless function
    replaced = replace_usefulness(df, question_stem)
    
    #Add up all the values, sort them
    normed = replaced.sum().sort_values(ascending=False)
   
    #Remove the question stem from each row index, leaving only the unique sub-question text
    normed.index=[s[len(question_stem):] for s in normed.index]
    
    #Drop some of the last ones if needed
    if drop_last != None: 
        normed.drop(normed.index[-1*drop_last:], inplace=True)
    
    plt.figure(figsize=figsize)
    ax = sns.barplot(y = normed.index, x = normed)
    plt.setp(ax.get_yticklabels(), fontsize=15)
    plt.title(question_stem + ', N = ' + str(len(replaced)))
    plt.xlabel('Usefulness')
    plt.show()
    
    return normed

#Next two function repeat previous two but for frequency instead of usefulness. 
def replace_frequency(df, question_stem):
    """Takes a DataFrame and a question stem, plots a scaled graph about usefulness.
    :param df: DataFrame
    :question_stem: a string containing the quesiton stem"""
    
    filtered = pd.DataFrame(df.loc[:,df.columns.str.startswith(question_stem)])
    filtered.replace({
        'Most of the time' : 1,
        'Often' : 0.6,
        'Sometimes' : 0.25,
        'Rarely' : 0.1,
        #np.nan : 0
    }, inplace=True)
    
    return filtered

def plot_frequency_questions(df, question_stem, figsize=(12,36), drop_last=None):
    """Plots a scaled frequency chart for multiple, related questions about frequency of use.
    :param df: dataframe
    :param question_stem: a string containing the question stem
    :param figsize: tuple for figure size, default = (12,36)"""
    
    replaced = replace_frequency(df, question_stem)
    normed = replaced.sum().sort_values(ascending=False)
    
    #for i in normed.index:
        #normed[i] = normed[i]/replaced[i].count()
    
                            
    normed.index=[s[len(question_stem):] for s in normed.index]
    if drop_last != None: 
        normed.drop(normed.index[-1*drop_last:], inplace=True)
    
    plt.figure(figsize=figsize)
    ax = sns.barplot(y = normed.index, x = normed)
    plt.setp(ax.get_yticklabels(), fontsize=15)
    plt.title(question_stem + ', N = ' + str(len(replaced)))
    plt.xlabel('Raw Score')
    plt.show()
    
    return normed

#Repeat previous function but for importance.
def replace_importance(df, question_stem):
    """Takes a DataFrame and a question stem. Plots a scaled graph about importance.
    :param df: DataFrame
    :question_stem: a string containing the quesiton stem"""
    
    filtered = pd.DataFrame(df.loc[:,df.columns.str.startswith(question_stem)])
    
    if 'Necessary' in filtered.values:
        replacements = {
            'Necessary' : 1,
            'Nice to have': 0.5,
            'Unnecessary' : 0,
            np.nan : 0
        }
    else:
        replacements = {
            'Very Important' : 1,
            'Somewhat important' : 0.5,
            'Not important' : 0,
            np.nan : 0
        }
    
    filtered.replace(replacements, inplace=True)
    
    return filtered

def plot_importance_questions(df, question_stem, figsize=(12,36), drop_last=None):
    """Plots a scaled frequency chart for multiple, related questions about frequency of use.
    :param df: dataframe
    :param question_stem: a string containing the question stem
    :param figsize: tuple for figure size, default = (12,36)"""
    
    replaced = replace_importance(df, question_stem)
    normed = replaced.sum().sort_values(ascending=False)
                            
    normed.index=[s[len(question_stem):] for s in normed.index]
    if drop_last != None: 
        normed.drop(normed.index[-1*drop_last:], inplace=True)
    
    plt.figure(figsize=figsize)
    ax = sns.barplot(y = normed.index, x = normed)
    plt.setp(ax.get_yticklabels(), fontsize=15)
    plt.title(question_stem + ', N = ' + str(len(replaced)))
    plt.xlabel('Importance')
    plt.show()
    
    return normed


#Importing the multiple choice responses
MC = pd.read_csv('../input/multipleChoiceResponses.csv', encoding='latin-1', low_memory=False)

#Originally I was going to examine the free reponse quesitons as well but decided to leave it for another time.
#FF = pd.read_csv('freeformResponses.csv',encoding='latin-1', low_memory=False)
#Creates a histogram of the Age column of the multiple choice DataFrame (called MC). The 'rwidth' just changes bar width
#to add a little aesthetic space.

MC.Age.hist(bins=20, figsize=(12,6), rwidth=0.85)
plt.axvline(x=MC.Age.median(), color='black', linestyle='--', label='Median: ' + str(MC.Age.median()) + ' years')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age of Survey Respondents N = ' + str(MC.Age.count()))
plt.legend(loc='upper right')
plt.show()

fraction_under_40 = MC.Age[MC.Age < 40].count() / len(MC.Age)
print("Fraction under 40 years old = " + str(fraction_under_40))
#Isolating dollars
money = MC.CompensationAmount[MC.CompensationCurrency == 'USD']

#Cleaning the data. Some had commas, had to drop the non-answers, and they were all recorded as strings. Also
#some people thought it would be cute to enter 99999999999.
money = money.str.replace(',', '')
money.dropna(inplace=True)
money = pd.to_numeric(money, errors='coerce')
money.sort_values(inplace=True)
money.drop([3013,5939], inplace=True)

#Data is all cleaned, but some people were making half a million dollars, which skewed the chart a lot. So I only graphed
#those making $300K or less. Original amounts are included in the median calculation.
money_median = money.median()
money_less = money[money <= 300000]

#Making the plot
money_less.hist(bins=30, histtype='bar', figsize=(12,6), rwidth=0.85)
plt.axvline(x=money_median, linestyle='--', color='darkred', label='Median = $' + str(money_median))
plt.xlabel('Salary in USD')
plt.ylabel('Count')
plt.title('Compensation in USD, N = ' + str(len(money)))
plt.legend(loc = 'upper right');
plt.figure(figsize=(12,6))
plt.title('Tenure Writing Data Code, N = ' + str(MC.Tenure.count()))

#A countplot is like a histogram for a discrete range. It just goes through a column, keeping a running tab of the values it finds.
sns.countplot(data=MC, y='Tenure', order=['Less than a year', '1 to 2 years', '3 to 5 years', '6 to 10 years', 'More than 10 years', 'I don\'t write code to analyze data']);
plt.ylabel('');
#The actual answers are below in red. Originally I was interested in a linear regression of the data, so I used the values
#below to replace the text answers. In the end, I thought the boxplots gave a better impression of the data, but I left
#replacement values there for future use.

replacements = {
    "I don't write code to analyze data" : np.nan,
    "1 to 2 years" : 2,
    "3 to 5 years" : 4,
    "6 to 10 years" : 8,
    "Less than a year" : 0.5,
    "More than 10 years": 11
}
x_tenure = MC.Tenure.replace(replacements)[money_less.index].dropna()
y_money = money_less[x_tenure.index]
plt.figure(figsize=(12,6))
sns.boxplot(x_tenure, y_money)
plt.ylim(0, 400000)
plt.title('Compensation vs. Tenure, N = ' + str(len(money)))
plt.show();
plt.figure(figsize=(12,6))
ax = sns.countplot(y='FormalEducation', data=MC, order=MC['FormalEducation'].value_counts().index)
plt.setp(ax.get_yticklabels(), fontsize=15)
plt.ylabel('')
plt.title('Level of Formal Education, N = ' + str(MC.FormalEducation.count()))
plt.show()

#This adds up the number of people who have a Masters and a PhD, then divides by the total.
fraction_higher_bachelors = (MC['FormalEducation'].value_counts()['Master\'s degree']
 + MC['FormalEducation'].value_counts()['Doctoral degree'])/len(MC.FormalEducation)
print('Fraction of respondents with more than a Bachelor\'s degree: ' + str(fraction_higher_bachelors))
plt.figure(figsize=(12,6))
plt.title('First Data Science Training, N = ' + str(MC.FirstTrainingSelect.count()))
ax = sns.countplot(y='FirstTrainingSelect', data=MC, order=MC['FirstTrainingSelect'].value_counts().index)
plt.setp(ax.get_yticklabels(), fontsize=15)
plt.ylabel('');
plt.show()
#Here I am using one of the functions I wrote in the first block of code. It uses some numeric replacements for text responses to get
#a good idea of the overall usefulness of a platform.
sns.set_palette(sns.cubehelix_palette(20, start=1.7, reverse=True))
plot_usefulness_questions(MC, 'LearningPlatformUsefulness', figsize=(10,10));
#This is the primary question I wrote the histogram function above. I thought there might be another questions to use it with, but it wasn't as useful.
filt = multi_plot_hist(MC, 'LearningCategory', figsize=(20,12));
plt.figure(figsize=(12,6))
plt.title('Most Important Way to Prove Knowledge, N = ' + str(MC.ProveKnowledgeSelect.count()))
ax = sns.countplot(data=MC, y='ProveKnowledgeSelect', order=MC['ProveKnowledgeSelect'].value_counts().index)
plt.setp(ax.get_yticklabels(), fontsize=15)
plt.ylabel('');
sns.set_palette(sns.cubehelix_palette(20, start=1.7, reverse=True))
plt.figure(figsize=(14,6))
plt.title('Industry, N = ' + str(len(MC.EmployerIndustry)))
ax = sns.countplot(y='EmployerIndustry', data=MC, order=MC['EmployerIndustry'].value_counts().index)

#This is the only way I figured out how to change the labels on the y-axis.
plt.setp(ax.get_yticklabels(), fontsize=13)
plt.ylabel('')
plt.show()
plt.figure(figsize=(8,8))
plt.title('Job Function')
ax = sns.countplot(y='JobFunctionSelect', data=MC, order=MC['JobFunctionSelect'].value_counts().index)
plt.setp(ax.get_yticklabels(), fontsize=13)
plt.ylabel('');
sns.set_palette(sns.cubehelix_palette(30, start=1.7, reverse=True))
plot_frequency_questions(MC, 'WorkToolsFrequency', figsize=(10,8), drop_last=25);
sns.set_palette(sns.cubehelix_palette(35, start=1.7, reverse=True))
plot_frequency_questions(MC, 'WorkMethodsFrequency', figsize=(10,8), drop_last = 12);
df = plot_importance_questions(MC, 'JobSkillImportance', figsize=(10,7), drop_last=3)
plt.figure(figsize=(12,6))
plt.title('Recommended Language')
sns.set_palette(sns.cubehelix_palette(13, start=1.7, reverse=True))
ax = sns.countplot(data=MC, y='LanguageRecommendationSelect', order=MC['LanguageRecommendationSelect'].value_counts().index);
plt.ylabel('')
plt.setp(ax.get_yticklabels(), fontsize=14);
plt.figure(figsize=(12,6))
plt.title('Job Search')
ax = sns.countplot(y='JobSearchResource', data=MC, order=MC['JobSearchResource'].value_counts().index)
plt.setp(ax.get_yticklabels(), fontsize=15)
plt.ylabel('');
sns.set_palette(sns.cubehelix_palette(13, start=1.7, reverse=True))
select_all_that_apply_plot(MC, 'MLSkillsSelect', figsize=(10,8))
plt.figure(figsize=(10,6))
plt.title('Level of Algorithum Understanding')
ax = sns.countplot(y='AlgorithmUnderstandingLevel', data=MC, order=MC['AlgorithmUnderstandingLevel'].value_counts().index);
plt.setp(ax.get_yticklabels(), fontsize=13)
plt.ylabel('');
sns.set_palette(sns.cubehelix_palette(30, start=1.7, reverse=True))
plot_frequency_questions(MC, 'WorkChallengeFrequency', figsize=(12,8), drop_last=6);
sns.set_palette(sns.cubehelix_palette(20, start=1.7, reverse=True))
plot_importance_questions(MC, 'JobFactor', figsize=(12,8));
plt.figure(figsize=(8,6))
plt.title('Data Scientist')
ax = sns.countplot(y='DataScienceIdentitySelect', data=MC, order=MC['DataScienceIdentitySelect'].value_counts().index);
plt.setp(ax.get_yticklabels(), fontsize=15)
plt.ylabel('');
plt.figure(figsize=(14,6))
sns.set_palette(sns.cubehelix_palette(12, start=1.7, reverse=True))
plt.title('Machine Learning at Work, N = ' + str(MC.EmployerMLTime.count()))
ax = sns.countplot(y='EmployerMLTime', data=MC, order=[
    'Less than one year',
    '1-2 years',
    '3-5 years',
    '6-10 years',
    'More than 10 years',
    'Don\'t know'
]);
plt.ylabel('')
plt.setp(ax.get_yticklabels(), fontsize=15);
plt.figure(figsize=(14,7))
plt.title('Job Satisfaction, N = ' + str(MC.JobSatisfaction.count()))
plt.ylabel('')
sns.countplot(x='JobSatisfaction', data=MC,
     order=[
         '1 - Highly Dissatisfied',
         '2',
         '3',
         '4',
         '5',
         '6',
         '7',
         '8',
         '9',
         '10 - Highly Satisfied',
     ]);
