import numpy as np

import pandas as pd



complete_survey = pd.read_csv("../input/stack-overflow-developer-survey-results-2019/survey_results_public.csv")

complete_survey_schema = pd.read_csv("../input/stack-overflow-developer-survey-results-2019/survey_results_schema.csv")
survey = complete_survey[[

    'MainBranch',

    'Hobbyist',

    'OpenSourcer',

    'Employment',

    'Country',

    'Student',

    'EdLevel',

    'UndergradMajor',

    'DevType',

    'YearsCode',

    'Age1stCode',

    'YearsCodePro',

    'ConvertedComp',

    'LanguageWorkedWith',

    'Age',

    'Gender'

]]
import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

# Enable inline usage of matplotlib within the notebook. See for more information:

# https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-matplotlib 

%matplotlib inline



x1 = np.linspace(-2*np.pi, 2*np.pi, 50)

y1 = np.sin(x1)

plt.plot(x1, y1)

plt.show()
x2 = np.linspace(-2*np.pi, 2*np.pi, 50)

plt.plot(x2, np.sin(x2))

plt.plot(x2, np.cos(x2))

plt.show()
x3 = np.linspace(-2*np.pi, 2*np.pi, 50)

plt.plot(x3, np.sin(x3), label='sin(x)') # Notice the additional label argument.

plt.plot(x3, np.cos(x3), label='cos(x)') # Notice the additional label argument.

plt.legend() # You need this call for the legends to show.

plt.show()
x4 = np.linspace(-2*np.pi, 2*np.pi, 50)

plt.plot(x4, np.sin(x4), label='sin(x)')

plt.plot(x4, np.cos(x4), label='cos(x)')

plt.legend()

plt.title("Comparison of sine and cosine") # Add a title to the graph

plt.xlabel("x")                            # Add a label to the x-axis

plt.ylabel("sin(x)\ncos(x)")               # Add a label to the y-axis

plt.grid(True)                             # Enable grid

plt.show()
x5 = np.linspace(-2*np.pi, 2*np.pi, 50)



# Notice the additional 'linestyle' and 'color' args.

plt.plot(x5, np.sin(x5), linestyle='-', color='r', label='sin(x)')

plt.plot(x5, np.cos(x5), linestyle='--', color='g', label='cos(x)')

plt.plot(x5, x5**2, linestyle=':', color='b', label='x^2')          # Added the x-squared function.

plt.plot(x5, np.exp(x5), linestyle='-.', color='y', label='cos(x)') # Added the exp(x) function.



plt.legend()

plt.title("Comparison of sine, cosine, x^2, and exp(x)")

plt.xlabel("x")

plt.ylabel("""sin(x)

cos(x)

x^2

exp(x)""")

plt.grid(True)

plt.show()
plt.figure(figsize=(10, 8))              # Made the graph larger as we have more functions now.

                                         # Notice that this call has to come before the calls to

                                         # the plot() methods, as this call instructs matplotlib

                                         # to start a new plot.

plt.xlim(-2*np.pi, 2*np.pi)              # Manually specifying the limits of the x and y axes.

plt.ylim(-5, 5)                          # This is necessary because the x^2 and the exp(x)

                                         # functions grow large quickly, making the sine and

                                         # cosine functions hard to notice.



# Notice the additional 'linestyle' and 'color' args.

plt.plot(x5, np.sin(x5), linestyle='-', color='r', label='sin(x)')

plt.plot(x5, np.cos(x5), linestyle='--', color='g', label='cos(x)')

plt.plot(x5, x5**2, linestyle=':', color='b', label='x^2')

plt.plot(x5, np.exp(x5), linestyle='-.', color='y', label='cos(x)')





plt.legend()

plt.title("Comparison of sine, cosine, x^2, and exp(x)")

plt.xlabel("x")

plt.ylabel("""sin(x)

cos(x)

x^2

exp(x)""")

plt.grid(True)

plt.show()
x6 = np.linspace(-2*np.pi, 2*np.pi, 50)

plt.figure(figsize=(10, 8))              # Made the graph larger as we have more functions now.

                                         # Notice that this call has to come before the calls to

                                         # the plot() methods, as this call instructs matplotlib

                                         # to start a new plot.



# Notice the additional 'linestyle' and 'color' args.

plt.plot(x6, np.sin(x6), '.g', label='sin(x)')



plt.legend()

plt.title("Graph of sin(x)")

plt.xlabel("x")

plt.ylabel("sin(x)")

plt.grid(True)

plt.show()
plt.bar([1, 2, 3, 4], [10, 20, 30, 40])
survey_with_age = survey.dropna(subset=['Age']) # First, drop rows which don't have a value in Age

survey_by_age = survey_with_age.groupby(pd.cut(survey_with_age['Age'], np.arange(0, 101, 10)))['Age'].count()

survey_by_age
cats = list(map(lambda x: str(x), survey_by_age.index))

values = survey_by_age.values

plt.figure(figsize=(14, 8))

plt.bar(cats, values)

survey['OpenSourcer'].unique()

top_countries = survey.groupby('Country')['Country'].count().sort_values(ascending=False).head(10)

top_countries
# Notice that we use .index to extract the country names, e.g. United States.

top_countries_names = top_countries.index

survey_top_countries = survey[survey['Country'].isin(top_countries_names)]



# verify that we indeed only has those countries.

survey_top_countries['Country'].unique()
survey_top_countries_active_in_os = survey_top_countries[survey_top_countries['OpenSourcer'] == 'Once a month or more often']

survey_top_countries_inactive_in_os = survey_top_countries[survey_top_countries['OpenSourcer'] != 'Once a month or more often']



bar1_heights = survey_top_countries_active_in_os.groupby('Country')['Country'].count()[top_countries_names]

bar2_heights = survey_top_countries_inactive_in_os.groupby('Country')['Country'].count()[top_countries_names]



plt.figure(figsize=(14, 8))

plt.bar(top_countries_names, bar1_heights, color='orange')

plt.bar(top_countries_names, bar2_heights, color='blue')

plt.figure(figsize=(14, 8))

plt.bar(top_countries_names, bar2_heights, color='blue')

plt.bar(top_countries_names, bar1_heights, color='orange')

plt.figure(figsize=(14, 8))



width = 0.4 # A width of 1.0 spans the whole area between two consecutive

            # tickts in the x-axis. Since we want to show two bars at each

            # tick, the width should be less than 0.5 for each bar so bars

            # from different tickts don't touch or overlap each other

x = np.arange(len(top_countries_names))

plt.bar(x - width/2, bar2_heights, width=width, color='blue') # left-shifted

plt.bar(x + width/2, bar1_heights, width=width, color='orange') # right-shifted



# Set the x-ticks and their labels.

ax = plt.axes()

ax.set_xticks(x)

ax.set_xticklabels(top_countries_names)



plt.show()
plt.figure(figsize=(14, 8))



width = 0.4

x = np.arange(len(top_countries_names))

plt.bar(x - width/2, bar2_heights, width=width, color='blue',

        label="Not actively contributing to open source") # Add label

plt.bar(x + width/2, bar1_heights, width=width, color='orange',

        label="Actively contributing to open source") # Add label



# Set the x-ticks and their labels.

ax = plt.axes()

ax.set_xticks(x)

ax.set_xticklabels(top_countries_names)

plt.legend() # Enable legends



plt.show()

survey['EdLevel'].unique()

survey_by_edlevel = survey.groupby('EdLevel')['EdLevel'].count().sort_values()

survey_by_edlevel = survey_by_edlevel * 100 / survey_by_edlevel.sum() # convert to percentages



labels = list(map(lambda x: str(x), survey_by_edlevel.index))

values = survey_by_edlevel.values



plt.figure(figsize=(14, 8))

plt.pie(values, labels=labels,

        explode=[0.1] * len(labels), # if the values here are non-zero, the slices of

                                     # the pie chart are moved away from the centre.

        autopct='%1.1f%%')           # tell matplotlib to print the percentages on the slices.

heatmap = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],

                    [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],

                    [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],

                    [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],

                    [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],

                    [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],

                    [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])





fig, ax = plt.subplots()

fig.set_size_inches(6, 6)

im = ax.imshow(heatmap)
import seaborn as sns;

plt.figure(figsize=(7, 6)) # increased the width to 7 compared to the

                           # previous code to account for the color bar

ax = sns.heatmap(heatmap)

import seaborn as sns;

plt.figure(figsize=(7, 6))

ax = sns.heatmap(heatmap, annot=True)

list(survey_top_countries['OpenSourcer'].unique()) # Converting to list to make it easier to read
def map_os(value):

    return {

      'Never': 0,

      'Less than once per year': 0,

      'Less than once a month but more than once per year': 6,

      'Once a month or more often': 12,

    }[value]

list(survey_top_countries['YearsCode'].unique()) # Converting to list to make it easier to read
def to_int(value):

    try:

        return int(value)

    except:

        return None


survey_temp = survey_top_countries[['Country', 'YearsCode', 'OpenSourcer']].copy()

survey_temp['OpenSourcer'] = survey_temp['OpenSourcer'].transform(lambda x: map_os(x))

survey_temp['YearsCode'] = survey_temp['YearsCode'].transform(lambda x: to_int(x))

survey_temp = survey_temp[(survey_temp['YearsCode'] >= 1) & (survey_temp['YearsCode'] <= 20)]



ptable = survey_temp.pivot_table(

    index='Country',

    columns='YearsCode',

    values='OpenSourcer'

)



plt.figure(figsize=(14, 6))

sns.heatmap(ptable, annot=True)
x1 = np.linspace(-2*np.pi, 2*np.pi, 200)

y1 = np.sin(x1)



plt.figure(figsize=(10, 6))



for i in range(1, 10):

    plt.subplot(3, 3, i)

    #plt.plot(x1, y1)

    plt.plot(x1, np.sin(i*x1))

plt.show()