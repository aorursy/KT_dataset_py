'''
Goal: determine the principal factors that affect program completion in 4-year colleges

This is Python implementation of my earlier study using R and RMarkdown:

https://www.kaggle.com/lkashif/d/kaggle/college-scorecard/mvas-to-predict-college-completion-rate
https://www.kaggle.com/lkashif/d/kaggle/college-scorecard/mvas-to-predict-college-completion-rmd

I use multivariate machine learning techniques, which I also refer to as MVAs. I train them to 
predict the fraction of students who complete the 4-year program in less than 6 years. This is 
therefore a regression problem. I use college and aggregated student data from 2011 to train 3 
different MVA methods:

    single decision tree
    random forest of decision trees
    support vector machine (SVM)

The learning is subsequently evaluated on 2013 data. Below are details of the workflow.

The variable that I want to predict (variable of interest VOI) is C150_4 in the DB. In the first 
step, I look at all quantitative variables associated with colleges and (aggregated) students. 14 
variables look promising in terms of predictive power:

    rate of admission for all demographics (ADM_RATE_ALL in the DB)
    total cost of attendance per year (COSTT4_A)
    fraction of students that are white (UGDS_WHITE)
    fraction of students that are black (UGDS_BLACK)
    fraction of part-time students (PPTUG_EF)
    fraction of students from low-income families, defined as annual family income < $30,000 (INC_PCT_LO)
    fraction of students above 25 years of age (UG25abv)
    fraction of first-generation college goers in the family (PAR_ED_PCT_1STGEN)
    fraction of students on federal loan (PCTFLOAN)
    75% percentile score on SAT reading (SATVR75)
    75% percentile score on SAT writing (SATWR75)
    75% percentile score on SAT math (SATMT75)
    tuition fee per year for in-state students (TUITIONFEE_IN)
    tuition fee per year for out-of-state students (TUITIONFEE_OUT)
'''

from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sqlite3
from sklearn import tree, svm, linear_model
from sklearn.ensemble import RandomForestRegressor

con = sqlite3.connect('../input/database.sqlite')
train = pd.read_sql("SELECT Year, ADM_RATE_ALL, COSTT4_A, UGDS_BLACK, PPTUG_EF, INC_PCT_LO, UG25abv, PAR_ED_PCT_1STGEN, PCTFLOAN, C150_4 \
FROM Scorecard \
WHERE Year = 2011 \
AND COSTT4_A != 'PrivacySuppressed' AND COSTT4_A IS NOT NULL \
AND ADM_RATE_ALL != 'PrivacySuppressed' AND ADM_RATE_ALL IS NOT NULL \
AND UGDS_BLACK != 'PrivacySuppressed' AND UGDS_BLACK IS NOT NULL \
AND PPTUG_EF != 'PrivacySuppressed' AND PPTUG_EF IS NOT NULL \
AND INC_PCT_LO != 'PrivacySuppressed' AND INC_PCT_LO IS NOT NULL \
AND UG25abv != 'PrivacySuppressed' AND UG25abv IS NOT NULL \
AND PAR_ED_PCT_1STGEN != 'PrivacySuppressed' AND PAR_ED_PCT_1STGEN IS NOT NULL \
AND PCTFLOAN != 'PrivacySuppressed' AND PCTFLOAN IS NOT NULL \
AND C150_4 != 'PrivacySuppressed' AND C150_4 IS NOT NULL", con)
'''
I collect these variables in a Pandas data frame and examine them for availability, integrity and 
correlations. The SAT scores are privacy-suppressed for the majority of cases, yielding <500 
records in 2011. So I decide to leave them out. Tuition fee per year for both in-state and 
out-of-state students is highly correlated with the total cost of attendance per year, as expected 
(correlation coefficients > 90%), so the tuition fee variables are excluded. Finally, the fraction 
of white students has a very small correlation with the VOI, so that this is excluded as well. 
That leaves me with 8 training variables and 1143 events to train with.
'''

plt.scatter(train.ADM_RATE_ALL, train.C150_4)
plt.xlabel('Fraction of students admitted')
plt.ylabel('Fraction of students completing program within 6 years')
plt.show()
plt.scatter(train.COSTT4_A, train.C150_4)
plt.xlabel('Cost of attendance/year')
plt.ylabel('Fraction of students completing program within 6 years')
plt.show()
plt.scatter(train.INC_PCT_LO, train.C150_4)
plt.xlabel('Fraction of students from families with income <$30000/yr')
plt.ylabel('Fraction of students completing program within 6 years')
plt.show()
plt.scatter(train.PAR_ED_PCT_1STGEN, train.C150_4)
plt.xlabel('Fraction of 1st-generation college students')
plt.ylabel('Fraction of students completing program within 6 years')
plt.show()
'''
I train a single decision tree first. The most discriminating variables are seen to be:

    fraction of students from low-income families (INC_PCT_LO)
    fraction of first-generation college goers in the family (PAR_ED_PCT_1STGEN)
    total cost of attendance per year (COSTT4_A)
    rate of admission (ADM_RATE_ALL)
    
I evaluate the performance on the 2013 data. I define a correct prediction when the normalized 
residual:

(predicted completion rate - actual rate)/actual rate

is smaller than 0.2, that is, the prediction of the learning algorithm is correct to within 20%.

The performance of a single tree is not very good: about 60% of predictions are correct as defined 
above.    
'''

test = pd.read_sql("SELECT Year, ADM_RATE_ALL, COSTT4_A, UGDS_BLACK, PPTUG_EF, INC_PCT_LO, UG25abv, PAR_ED_PCT_1STGEN, PCTFLOAN, C150_4 \
                    FROM Scorecard \
                    WHERE Year = 2013 \
                    AND COSTT4_A != 'PrivacySuppressed' AND COSTT4_A IS NOT NULL \
                    AND ADM_RATE_ALL != 'PrivacySuppressed' AND ADM_RATE_ALL IS NOT NULL \
                    AND UGDS_BLACK != 'PrivacySuppressed' AND UGDS_BLACK IS NOT NULL \
                    AND PPTUG_EF != 'PrivacySuppressed' AND PPTUG_EF IS NOT NULL \
                    AND INC_PCT_LO != 'PrivacySuppressed' AND INC_PCT_LO IS NOT NULL \
                    AND UG25abv != 'PrivacySuppressed' AND UG25abv IS NOT NULL \
                    AND PAR_ED_PCT_1STGEN != 'PrivacySuppressed' AND PAR_ED_PCT_1STGEN IS NOT NULL \
                    AND PCTFLOAN != 'PrivacySuppressed' AND PCTFLOAN IS NOT NULL \
                    AND C150_4 != 'PrivacySuppressed' AND C150_4 IS NOT NULL", con)

target = train["C150_4"].values
features = train[["ADM_RATE_ALL", "COSTT4_A", "UGDS_BLACK", "PPTUG_EF",  "INC_PCT_LO", "UG25abv", "PAR_ED_PCT_1STGEN", "PCTFLOAN"]].values
features_test = test[["ADM_RATE_ALL", "COSTT4_A", "UGDS_BLACK", "PPTUG_EF",  "INC_PCT_LO", "UG25abv", "PAR_ED_PCT_1STGEN", "PCTFLOAN"]].values

model_tree = tree.DecisionTreeRegressor()
model_tree = model_tree.fit(features, target)
pred_tree = model_tree.predict(features_test)
accu = abs(pred_tree - test.C150_4)/test.C150_4 < 0.2
frac = sum(accu)/len(accu)
print('Accuracy of single tree: ' + str(frac))
'''
I then train a random forest of trees using the default parameters. The four highest ranked 
variables are the same as in the case of the single tree. The prediction accuracy is better in this
case: over 70%.
'''

model_forest = RandomForestRegressor()
model_forest = model_forest.fit(features, target)
pred_forest = model_forest.predict(features_test)
accu = abs(pred_forest - test.C150_4)/test.C150_4 < 0.2
frac = sum(accu)/len(accu)
print('Accuracy of random forest: ' + str(frac))
'''
I also try linear regression using the same variables, and the result is acutally better than with
a single tree, and nearly as good as the forest.
'''

model_linear = linear_model.LinearRegression()
model_linear = model_linear.fit(features, target)
pred_linear = model_linear.predict(features_test)
accu = abs(pred_linear - test.C150_4)/test.C150_4 < 0.2
frac = sum(accu)/len(accu)
print('Accuracy of linear regression: ' + str(frac))
'''
The number of colleges as a function of the completion rate are shown from the test data, from the 
random forest prediction and from the linear fit prediction. Both methods perform badly for large values 
of the completion rate. The next step in the analysis would be to look at distributions of the training 
variables separately for low and high completion rates, and try to understand why the training is 
poorer in the latter case.
'''

plt.hist(test.C150_4)
plt.title('Number of colleges vs completion rate in test data')
plt.xlabel('Completion rate')
plt.ylabel('Number of colleges')
plt.show()
plt.hist(pred_forest)
plt.title('Number of colleges vs completion rate predicted by random forest')
plt.xlabel('Completion rate')
plt.ylabel('Number of colleges')
plt.show()
plt.hist(pred_linear)
plt.title('Number of colleges vs completion rate predicted by linear fit')
plt.xlabel('Completion rate')
plt.ylabel('Number of colleges')
plt.show()
'''
The data are not very reliable owing to the large number of privacy suppressions. In any case,
the important conclusions of this study are not the performance of the machine learning method, 
but the factors that affect program completion rates the most. In general, they agree with our 
naive expectations as to why someone may fail to complete a college program.
'''