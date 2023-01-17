import csv, sys, math

import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

%matplotlib inline



# Read data into a pandas dataframe

einsteinFile = "/kaggle/input/uncover/einstein/diagnosis-of-covid-19-and-its-clinical-spectrum.csv"

healthDF = pd.read_csv(einsteinFile)



healthDF.head()
# Show the columns available in this dataset

healthDF.columns.values
columns = list(healthDF.columns)

labTests = columns[6:]



minSampleSize = 360



# Count the number of recordings for each lab test. Disallow test if the count is below minSampleSize.

usableLabTests = list()



for labTest in labTests:

    sampleCount = len(healthDF[healthDF[labTest].notnull()])

    

    if sampleCount >= minSampleSize:

        usableLabTests.append(labTest)

        

print("__Lab tests with sample size >= " + str(minSampleSize) + "__\n" + str(usableLabTests))
floatType = np.dtype('float64')



numericalLabTests = list()

booleanLabTests = list()



for test in usableLabTests:

    if healthDF[test].dtype == floatType:

        numericalLabTests.append(test)        

    else:

        uniqueVals = list(healthDF[test].unique())



        # Look for present of the word "detected", since there is at least one test where "not_detected" is 

        # the only non-null result (not useful).

        if "detected" in uniqueVals:

            booleanLabTests.append(test)



print("Numerical tests: " + str(numericalLabTests))

print("\nBoolean tests: " + str(booleanLabTests))
print("Original number of patients: " + str(len(healthDF)))

patientsWithNumericalResults = list()



for index, row in healthDF.iterrows():

    patientID = row['patient_id']

    

    hasResults = False

    for numTest in numericalLabTests:

        if not pd.isna(row[numTest]):

            hasResults = True

            break

    

    if hasResults:

        patientsWithNumericalResults.append(patientID)

        

patientsWithResultsDF = healthDF[healthDF['patient_id'].isin(patientsWithNumericalResults)].copy()



print("Number of patients with at least one numerical lab result: " + str(len(patientsWithResultsDF)))



testStats = dict()

minErrorToTest = dict()



for test in numericalLabTests:

    # Calculate mean, stdev

    testStats[test] = dict()

    

    meanVal = patientsWithResultsDF[test].mean()

    minVal = patientsWithResultsDF[test].min()

    maxVal = patientsWithResultsDF[test].max()

    stdVal = patientsWithResultsDF[test].std()



    testStats[test]["mean"] = meanVal

    testStats[test]["std"] = stdVal

    testStats[test]["min"] = minVal

    testStats[test]["max"] = maxVal

    testStats[test]["min1Std"] = meanVal - stdVal

    testStats[test]["max1Std"] = meanVal + stdVal

     

    minError = meanVal - minVal

    if minError not in minErrorToTest:

        minErrorToTest[minError] = list()

        

    minErrorToTest[minError].append(test)



sortedMins = reversed(sorted(minErrorToTest.keys()))



meansForPlot = list()

testNamesForPlot = list()

minsForPlot = list()

maxsForPlot = list()

minStdForPlot = list()

maxStdForPlot = list()



for sm in sortedMins:

    for testName in minErrorToTest[sm]:

        minsForPlot.append(sm)

        testNamesForPlot.append(testName)

        meansForPlot.append(testStats[testName]["mean"])

        maxsForPlot.append(testStats[testName]["max"])

        

        minStdForPlot.append(testStats[testName]["min1Std"])

        maxStdForPlot.append(testStats[testName]["max1Std"])



fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True)

ax0.plot(testNamesForPlot[0:10], minStdForPlot[0:10], 'b--', label='1 std dev')

ax0.plot(testNamesForPlot[0:10], maxStdForPlot[0:10], 'b--')

ax0.errorbar(testNamesForPlot[0:10], meansForPlot[0:10], yerr=[minsForPlot[0:10], maxsForPlot[0:10]], fmt='o', label='test Mean')



ax1.plot(testNamesForPlot[10:], minStdForPlot[10:], 'b--', label='1 std dev')

ax1.plot(testNamesForPlot[10:], maxStdForPlot[10:], 'b--')

ax1.errorbar(testNamesForPlot[10:], meansForPlot[10:], yerr=[minsForPlot[10:], maxsForPlot[10:]], fmt='o', label='test Mean')



ax0.tick_params(labelsize=10)

ax1.tick_params(labelsize=10)



ax0.legend(loc='upper left')

ax1.legend(loc='upper left')



plt.title("Mean and min/max range for numerical lab tests")

plt.setp(ax0.xaxis.get_majorticklabels(), rotation=90)

plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)

plt.figure(figsize=(50,50))

plt.tight_layout()

plt.show()
def calculateHighLowNormalTest(row, mode, testNames, testStats):

    

    testCount = 0

    

    for testName in testNames:

        stats = testStats[testName]       

    

        if mode == "high":

            if row[testName] > stats["max1Std"]:

                testCount += 1

                

        elif mode == "low":

            if row[testName] < stats["min1Std"]:

                testCount += 1

                

        elif mode == "normal":

            if row[testName] <= stats["max1Std"] and row[testName] >= stats["min1Std"]:

                testCount += 1

        

        else:

            print("Unrecognized mode: " + mode)

            sys.exit(1)

    

    return testCount

    

def calculateBooleanTest(row, stringToFind, booleanTests):

    count = 0

    

    for bt in booleanTests:

        if isinstance(row[bt], float):

            continue

            

        if row[bt] == stringToFind:

            count += 1

            

    return count



    

def sumRecordedTests(row, testType, testNames):

    count = 0

    

    if testType == "num":

        for testName in testNames:

            if not math.isnan(row[testName]):

                count += 1

                

    elif testType == "bool":

        for testName in testNames:

            if isinstance(row[testName], float):

                continue

            if "detected" in row[testName]:

                count += 1

            

    return count



patientsWithResultsDF['numHighTests'] = patientsWithResultsDF.apply(lambda row: calculateHighLowNormalTest(row, "high", numericalLabTests, testStats), axis=1)

patientsWithResultsDF['numLowTests'] = patientsWithResultsDF.apply(lambda row: calculateHighLowNormalTest(row, "low", numericalLabTests, testStats), axis=1)

patientsWithResultsDF['numNormalTests'] = patientsWithResultsDF.apply(lambda row: calculateHighLowNormalTest(row, "normal", numericalLabTests, testStats), axis=1)



patientsWithResultsDF['numDetectedTests'] = patientsWithResultsDF.apply(lambda row: calculateBooleanTest(row, "detected", booleanLabTests), axis=1)

patientsWithResultsDF['numNotDetectedTests'] = patientsWithResultsDF.apply(lambda row: calculateBooleanTest(row, "not_detected", booleanLabTests), axis=1)



patientsWithResultsDF['numValidNumTests'] = patientsWithResultsDF.apply(lambda row: sumRecordedTests(row, "num", numericalLabTests), axis=1)

patientsWithResultsDF['numValidBoolTests'] = patientsWithResultsDF.apply(lambda row: sumRecordedTests(row, "bool", booleanLabTests), axis=1)
patientsWithResultsDF.head()
# Patients with abnormal test results (possible cancer)

group1 = patientsWithResultsDF[((patientsWithResultsDF['numHighTests'] + patientsWithResultsDF['numLowTests']) > 0) &

                                         (patientsWithResultsDF['numValidBoolTests'] > 0) & 

                                         (patientsWithResultsDF['numNotDetectedTests'] == patientsWithResultsDF['numValidBoolTests'])]# &



group1Size = len(group1)



positiveGroup1 = len(group1[group1['sars_cov_2_exam_result'] == 'positive'])

negativeGroup1 = len(group1[group1['sars_cov_2_exam_result'] == 'negative'])



print("Group 1 (Cancer patients) size: " + str(group1Size))
# Patients no high/low numerical tests (no cancer)

group2 = patientsWithResultsDF[(patientsWithResultsDF['numNormalTests'] > 0) &

                                ((patientsWithResultsDF['numHighTests'] + patientsWithResultsDF['numLowTests']) == 0)]



group2Size = len(group2)



positiveGroup2 = len(group2[group2['sars_cov_2_exam_result'] == 'positive'])

negativeGroup2 = len(group2[group2['sars_cov_2_exam_result'] == 'negative'])



print("Group 2 (Non-Cancer patients) size: " + str(group2Size))
print("Group 1 (Cancer) fraction of patients testing positive for COVID-19:\t" + str(positiveGroup1/group1Size))

print("Group 2 (No Cancer) fraction of patients testing positive for COVID-19:\t" + str(positiveGroup2/group2Size))

print()

print("Group 1 (Cancer) fraction of patients testing negative for COVID-19:\t" + str(negativeGroup1/group1Size))

print("Group 2 (No Cancer) fraction of patients testing negative for COVID-19:\t" + str(negativeGroup2/group2Size))
# Graphical representation of the numbers

labels = ['Pos for COVID-19', 'Neg for COVID-19']

group1Numbers = [positiveGroup1/group1Size, negativeGroup1/group1Size]

group2Numbers = [positiveGroup2/group2Size, negativeGroup2/group2Size]



x = np.arange(len(labels))

width = 0.35



fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2, group1Numbers, width, label='Group 1 (Cancer) - ' + str(group1Size) + ' Patients')

rects2 = ax.bar(x + width/2, group2Numbers, width, label='Group 2 (No Cancer) - ' + str(group2Size) + ' Patients')



ax.set_ylabel('Fraction')

ax.set_title('Coronavirus Diagnosis in Cancer vs. Non-Cancer Patients')

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend()

plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)



plt.show()
positiveGroup1_regularWard = len(group1[(group1['sars_cov_2_exam_result'] == 'positive') & (group1['patient_addmited_to_regular_ward_1_yes_0_no'] == 't')])

positiveGroup1_semiIU = len(group1[(group1['sars_cov_2_exam_result'] == 'positive') & (group1['patient_addmited_to_semi_intensive_unit_1_yes_0_no'] == 't')])

positiveGroup1_ICU = len(group1[(group1['sars_cov_2_exam_result'] == 'positive') & (group1['patient_addmited_to_intensive_care_unit_1_yes_0_no'] == 't')])

positiveGroup1_noWard = group1Size - positiveGroup1_regularWard - positiveGroup1_semiIU - positiveGroup1_ICU



negativeGroup1_regularWard = len(group1[(group1['sars_cov_2_exam_result'] == 'negative') & (group1['patient_addmited_to_regular_ward_1_yes_0_no'] == 't')])

negativeGroup1_semiIU = len(group1[(group1['sars_cov_2_exam_result'] == 'negative') & (group1['patient_addmited_to_semi_intensive_unit_1_yes_0_no'] == 't')])

negativeGroup1_ICU = len(group1[(group1['sars_cov_2_exam_result'] == 'negative') & (group1['patient_addmited_to_intensive_care_unit_1_yes_0_no'] == 't')])

negativeGroup1_noWard = group1Size - negativeGroup1_regularWard - negativeGroup1_semiIU - negativeGroup1_ICU



positiveGroup2_regularWard = len(group2[(group2['sars_cov_2_exam_result'] == 'positive') & (group2['patient_addmited_to_regular_ward_1_yes_0_no'] == 't')])

positiveGroup2_semiIU = len(group2[(group2['sars_cov_2_exam_result'] == 'positive') & (group2['patient_addmited_to_semi_intensive_unit_1_yes_0_no'] == 't')])

positiveGroup2_ICU = len(group2[(group2['sars_cov_2_exam_result'] == 'positive') & (group2['patient_addmited_to_intensive_care_unit_1_yes_0_no'] == 't')])

positiveGroup2_noWard = group2Size - positiveGroup2_regularWard - positiveGroup2_semiIU - positiveGroup2_ICU



negativeGroup2_regularWard = len(group2[(group2['sars_cov_2_exam_result'] == 'negative') & (group2['patient_addmited_to_regular_ward_1_yes_0_no'] == 't')])

negativeGroup2_semiIU = len(group2[(group2['sars_cov_2_exam_result'] == 'negative') & (group2['patient_addmited_to_semi_intensive_unit_1_yes_0_no'] == 't')])

negativeGroup2_ICU = len(group2[(group2['sars_cov_2_exam_result'] == 'negative') & (group2['patient_addmited_to_intensive_care_unit_1_yes_0_no'] == 't')])

negativeGroup2_noWard = group2Size - negativeGroup2_regularWard - negativeGroup2_semiIU - negativeGroup2_ICU
print("Group 1 (Cancer) COVID-positive patients admitted to the regular ward:\t\t" + str(positiveGroup1_regularWard/group1Size))

print("Group 2 (No Cancer) COVID-positive patients admitted to the regular ward:\t" + str(positiveGroup2_regularWard/group2Size))

print()

print("Group 1 (Cancer) COVID-positive patients admitted to the Semi-Intensive Unit:\t\t" + str(positiveGroup1_semiIU/group1Size))

print("Group 2 (No Cancer) COVID-positive patients admitted to the Semi-Intensive Unit:\t" + str(positiveGroup2_semiIU/group2Size))

print()

print("Group 1 (Cancer) COVID-positive patients admitted to the ICU:\t\t" + str(positiveGroup1_ICU/group1Size))

print("Group 2 (No Cancer) COVID-positive patients admitted to the ICU:\t" + str(positiveGroup2_ICU/group2Size))

print()

print("Group 1 (Cancer) COVID-positive patients not admitted to any ward:\t\t" + str(positiveGroup1_noWard/group1Size))

print("Group 2 (No Cancer) COVID-positive patients not admitted to any ward:\t" + str(positiveGroup2_noWard/group2Size))

print("----------")

print("Group 1 (Cancer) COVID-negative patients admitted to the regular ward:\t\t" + str(negativeGroup1_regularWard/group1Size))

print("Group 2 (No Cancer) COVID-negative patients admitted to the regular ward:\t" + str(negativeGroup2_regularWard/group2Size))

print()

print("Group 1 (Cancer) COVID-negative patients admitted to the Semi-Intensive Unit:\t\t" + str(negativeGroup1_semiIU/group1Size))

print("Group 2 (No Cancer) COVID-negative patients admitted to the Semi-Intensive Unit:\t" + str(negativeGroup2_semiIU/group2Size))

print()

print("Group 1 (Cancer) COVID-negative patients admitted to the ICU:\t\t" + str(negativeGroup1_ICU/group1Size))

print("Group 2 (No Cancer) COVID-negative patients admitted to the ICU:\t" + str(negativeGroup2_ICU/group2Size))

print()

print("Group 1 (Cancer) COVID-negative patients not admitted to any ward:\t\t" + str(negativeGroup1_noWard/group1Size))

print("Group 2 (No Cancer) COVID-negative patients not admitted to any ward:\t" + str(negativeGroup2_noWard/group2Size))
labels = ['Pos COVID-19 admitted to Regular Ward', 'Pos COVID-19 admitted to Semi-Intensive Ward', 

          'Pos COVID-19 admitted to ICU', 'Pos COVID-19 not admitted']

group1Numbers = [ positiveGroup1_regularWard/group1Size, positiveGroup1_semiIU/group1Size, 

                 positiveGroup1_ICU/group1Size, positiveGroup1_noWard/group1Size ]

group2Numbers = [ positiveGroup2_regularWard/group2Size, positiveGroup2_semiIU/group2Size, 

                 positiveGroup2_ICU/group2Size, positiveGroup2_noWard/group2Size ]



x = np.arange(len(labels))

width = 0.35



fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2, group1Numbers, width, label='Group 1 (Cancer) - ' + str(group1Size) + ' Patients')

rects2 = ax.bar(x + width/2, group2Numbers, width, label='Group 2 (No Cancer) - ' + str(group2Size) + ' Patients')



ax.set_ylabel('Fraction')

ax.set_title('Admittance for Positive COVID-19 Cancer vs. Non-Cancer Patients')

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend()

plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

plt.setp(ax.xaxis.get_majorticklabels(), rotation=80)



plt.show()



labels = ['Neg COVID-19 admitted to Regular Ward', 'Neg COVID-19 admitted to Semi-Intensive Ward', 

          'Neg COVID-19 admitted to ICU', 'Neg COVID-19 not admitted']

group1Numbers = [ negativeGroup1_regularWard/group1Size, negativeGroup1_semiIU/group1Size,

                 negativeGroup1_ICU/group1Size, negativeGroup1_noWard/group1Size ]

group2Numbers = [ negativeGroup2_regularWard/group2Size, negativeGroup2_semiIU/group2Size,

                 negativeGroup2_ICU/group2Size, negativeGroup2_noWard/group2Size ]



x = np.arange(len(labels))

width = 0.35



fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2, group1Numbers, width, label='Group 1 (Cancer) - ' + str(group1Size) + ' Patients')

rects2 = ax.bar(x + width/2, group2Numbers, width, label='Group 2 (No Cancer) - ' + str(group2Size) + ' Patients')



ax.set_ylabel('Fraction')

ax.set_title('Admittance for Negative COVID-19 Cancer vs. Non-Cancer Patients')

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend()

plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

plt.setp(ax.xaxis.get_majorticklabels(), rotation=80)



plt.show()


