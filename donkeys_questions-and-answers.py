# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_responses = pd.read_csv("/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv")

df_other_text = pd.read_csv("/kaggle/input/kaggle-survey-2019/other_text_responses.csv")
df_responses.shape
df_schema = pd.read_csv("/kaggle/input/kaggle-survey-2019/survey_schema.csv")
df_schema.shape
#df_schema
df_responses
from IPython.display import display, HTML



def print_full(x):

    pd.set_option('display.max_rows', len(x))

    pd.set_option('display.max_columns', None)

    pd.set_option('display.width', 2000)

    pd.set_option('display.float_format', '{:20,.2f}'.format)

    pd.set_option('display.max_colwidth', -1)

    x = x.style.set_properties(**{'text-align': 'left'})

    display(x)

#    print(x)

    pd.reset_option('display.max_rows')

    pd.reset_option('display.max_columns')

    pd.reset_option('display.width')

    pd.reset_option('display.float_format')

    pd.reset_option('display.max_colwidth')
qs = df_responses.iloc[0].T.to_frame()

qs.index.name = "name"

qs.columns = ["description"]
print_full(qs)
q_to_plot = [9, 12, 13, 14, 16, 17, 18, 20, 21, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

q_count_to_plot = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 15, 19, 22, 23]
df_responses = df_responses.fillna(0)
def create_df_q(qn):

    cols = [col for col in df_responses if f"Q{qn}_" in col]

    q_data = []

    names = []

    title = None

    for col in cols:

        parts = qs.loc[col]["description"].split("-")

        q_str = parts[0].split("?")[0]

        title = q_str

        name = parts[2].strip()

        names.append(name)

        count = df_responses[col].astype(bool).sum()

        q_data.append(count)

#    other_values = df_responses[cols[-1]]

    q_data = q_data[:-1]

    names = names[:-1]

#    others = pd.DataFrame(columns=["other"], data=other_values)

    df_x = pd.DataFrame(columns=names, data=[q_data])

    others = df_other_text[cols[-1]].str.lower().value_counts()

    return df_x, title, others
import matplotlib.pyplot as plt



def plot_q(qn):

    df_q, title, others = create_df_q(qn)

    ax = df_q.T.plot(kind="bar", figsize=(10,6))

    ax.set_title(title, fontsize=20)

    #ax.xaxis.label.set_size(20)

    ax.tick_params(axis="x", labelsize=20)

    plt.show()

    display(others.to_frame()[:10])

#plot_q(25)
for qn in q_to_plot:

    plot_q(qn)
df_responses = pd.read_csv("/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv")
orders = {

    1: None,

    2: None,

    3: None,

    4: ["No formal education past high school", "Professional degree", "Some college/university study without earning a bachelor’s degree", "Bachelor’s degree", "Master’s degree", "Doctoral degree", "I prefer not to answer"],

    5: None,

    6: ["0-49 employees", "50-249 employees", "250-999 employees", "1000-9,999 employees", "> 10,000 employees"],

    7: ["0", "1-2", "3-4", "5-9", "10-14", "15-19", "20+"],

    8: ["No (we do not use ML methods)", "We are exploring ML methods (and may one day put a model into production)", "We use ML methods for generating insights (but do not put working models into production)", "We recently started using ML methods (i.e., models in production for less than 2 years)", "We have well established ML methods (i.e., models in production for more than 2 years)", "I do not know"],

    10: ["$0-999", "1,000-1,999", "2,000-2,999", "3,000-3,999", "4,000-4,999", "5,000-7,499", "7,500-9,999", "10,000-14,999", "15,000-19,999", "20,000-24,999", "25,000-29,999", "30,000-39,999", "40,000-49,999", "50,000-59,999", "60,000-69,999", "70,000-79,999", "80,000-89,999", "90,000-99,999", "100,000-124,999", "125,000-149,999", "150,000-199,999", "200,000-249,999", "250,000-299,999", "300,000-500,000", "> $500,000"],

    11: ["$0 (USD)", "$1-$99", "$100-$999", "$1000-$9,999", "$10,000-$99,999", "> $100,000 ($USD)"],

    15: ["I have never written code", "< 1 years", "1-2 years", "3-5 years", "5-10 years", "10-20 years", "20+ years"],

    19: None,

    22: ["Never", "Once", "2-5 times", "6-24 times", "> 25 times"],

    23: ["< 1 years", "1-2 years", "2-3 years", "3-4 years", "4-5 years", "5-10 years", "10-15 years", "20+ years"]    

}
def plot_count_qs():

    for qn in q_count_to_plot:

        q_data = df_responses[f"Q{qn}"]

        title = q_data[0]

        counts = q_data[1:].dropna().value_counts()

        width = 8

        if len(counts) > 20:

            width = 24

        ordering = orders[qn]

        #print("using ordering:"+str(ordering))

        if ordering is None:

            plot_data = counts.sort_index()

        else:

            idx = pd.Categorical(counts.index.values,

                      categories=ordering,

                      ordered=True)

            counts = counts.reindex(idx)

            plot_data = counts.sort_index()

        ax = plot_data.plot(kind="bar", figsize=(width,6))

        ax.set_title(title, fontsize=20)

        ax.tick_params(axis="x", labelsize=20)

        #print(qn)

        plt.show()

        display(plot_data.to_frame())

plot_count_qs()