# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
## Defining common tools for IDA



## Data Preview Tools

def clean_from_null_empty(df, col):

    return df[col][~( (df[col].isnull()) | (df[col] == ''))]



def ida_util(df, details={}):

    total_rows = df.shape[0]

    ida_cols = []

    for col in df:

        cleaned_series = clean_from_null_empty(df, col)

        total_null = cleaned_series.count()

        col_detail = details.get("col", {

            "type": "",

            "description": "",

            "actionable": ""

        })

        ida_cols.append({

            "col": col,

            "totalNull": total_rows - total_null,

            "totalNotNull": total_null,

            "totalRows": total_rows,

            "totalUnique": cleaned_series.unique().shape[0],

            **col_detail

        })

    

    df_ida = pd.DataFrame(ida_cols)

    

    return df_ida
# Read data scripts

files = ['professionals.csv', 'groups.csv', 'comments.csv', 'school_memberships.csv', 'tags.csv', 'emails.csv', 'group_memberships.csv', 'answers.csv', 'students.csv', 'matches.csv', 'questions.csv', 'tag_users.csv', 'tag_questions.csv']

for file in files:

    print(file.replace(".csv", "") + " = " + "pd.read_csv(\"../input/{}\")".format(file))
professionals = pd.read_csv("../input/professionals.csv")

groups = pd.read_csv("../input/groups.csv")

comments = pd.read_csv("../input/comments.csv")

school_memberships = pd.read_csv("../input/school_memberships.csv")

tags = pd.read_csv("../input/tags.csv")

emails = pd.read_csv("../input/emails.csv")

group_memberships = pd.read_csv("../input/group_memberships.csv")

answers = pd.read_csv("../input/answers.csv")

students = pd.read_csv("../input/students.csv")

matches = pd.read_csv("../input/matches.csv")

questions = pd.read_csv("../input/questions.csv")

tag_users = pd.read_csv("../input/tag_users.csv")

tag_questions = pd.read_csv("../input/tag_questions.csv")
questions.head()
questions.describe()
answers.head()
answers.describe()
group_memberships.head()
group_memberships.describe()
# professionals["professionals_location"].notnull().sum()

# define null: null or NaN

# define a df: totalNull, totalNotNull, totalUniqueValuesExcludingNaN, type, description, actionable
# def clean_from_null_empty(df, col):

#     return df[col][~( (df[col].isnull()) | (df[col] == ''))]



# def ida_util(df, details={}):

#     total_rows = df.shape[0]

#     ida_cols = []

#     for col in df:

#         cleaned_series = clean_from_null_empty(df, col)

#         total_null = cleaned_series.count()

#         col_detail = details.get("col", {

#             "type": "",

#             "description": "",

#             "actionable": ""

#         })

#         ida_cols.append({

#             "col": col,

#             "totalNull": total_rows - total_null,

#             "totalNotNull": total_null,

#             "totalRows": total_rows,

#             "totalUnique": cleaned_series.unique().shape[0],

#             **col_detail

#         })

    

#     df_ida = pd.DataFrame(ida_cols)

    

#     return df_ida
ida_util(professionals)
df = pd.DataFrame({

    "a": [np.NaN, "", 1]

})

series = df["a"]

series = series[~( (series.isnull()) | (series == ''))]

series
professionals["professionals_location"].unique().shape
professionals.head()
professionals.describe()
groups.head()
groups.describe()
comments.head()
comments.describe()
school_memberships.head()
school_memberships.describe()
tags.head()
tags.describe()
emails.head()
emails.describe()
students.head()
students.describe()
matches.head()
matches.describe()
tag_users.head()
tag_users.describe()
tag_questions.head()
tag_questions.describe()