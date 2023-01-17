# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# read in our data

fiveDaySurveyData = pd.read_csv("../input/5day-data-challenge-signup-survey-responses/anonymous-survey-responses.csv")

kaggleSurveyData = pd.read_csv('../input/kaggle-survey-2017/multipleChoiceResponses.csv', encoding="ISO-8859-1")
import scipy.stats as sta # statistics

cs_data = kaggleSurveyData['CodeWriter'].value_counts()

print(sta.chisquare(cs_data))

prog_data = fiveDaySurveyData["Do you have any previous experience with programming?"].value_counts()

print(sta.chisquare(prog_data))



contingencyTable = pd.crosstab(fiveDaySurveyData["Do you have any previous experience with programming?"],

                               kaggleSurveyData['CodeWriter'])

           

sta.chi2_contingency(contingencyTable)