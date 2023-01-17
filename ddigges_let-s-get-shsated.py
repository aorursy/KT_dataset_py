import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
school_df = pd.read_csv("../input/2016 School Explorer.csv")
school_df.head()
school_df.dtypes
percentage_columns = ['Percent ELL', 'Percent Asian', 'Percent Black', 'Percent Hispanic', 'Percent Black / Hispanic', 'Percent White', 'Student Attendance Rate', 'Percent of Students Chronically Absent', 'Rigorous Instruction %', 'Collaborative Teachers %', 'Supportive Environment %', 'Effective School Leadership %', 'Strong Family-Community Ties %', 'Trust %']
def parse_percent(val):
    """
        If nan or empty string, return nan
        else remove percentage sign from the string if present
        and cast to integer
    """
    percent_sign = "%"
    if( pd.isnull(val) or len(val) == 0):
        return np.nan
    
    if (percent_sign in val):
        return float(val.replace(percent_sign, ""))
    
    return float(val.replace(percent_sign, ""))
school_df[percentage_columns] = school_df[percentage_columns].applymap(parse_percent)
school_df[percentage_columns].dtypes
def parse_income(val):
    """
        Parses a string representing an income by:
        1. Removing dollar and commas in the representation
        2. Returning the float value of the same
    """
    if( pd.isnull(val) or len(val) == 0):
        return np.nan
    
    val = re.sub('[$,]', '', val)
    
    return float(val)
school_df['School Income Estimate'] = school_df['School Income Estimate'].apply(parse_income)
school_df[school_df['Adjusted Grade'].notnull()]['Adjusted Grade']
school_df[school_df['New?'].notnull()]['New?']
school_df[school_df["Other Location Code in LCGMS"].notnull()]["Other Location Code in LCGMS"]
school_df = school_df.drop(["Adjusted Grade", "New?", "Other Location Code in LCGMS"], axis=1)
def is_public(val):
    """
        Returns true if the string val
        contains the substring 'P.S'
    """
    return 'P.S.' in val
school_df["Is public"] = school_df["School Name"].apply(is_public)
rating_columns = ['Rigorous Instruction Rating', 'Collaborative Teachers Rating', 'Supportive Environment Rating', 'Effective School Leadership Rating', 'Strong Family-Community Ties Rating', 'Trust Rating', 'Student Achievement Rating']
school_df[rating_columns] = school_df[rating_columns].apply(lambda x: x.astype('category'))