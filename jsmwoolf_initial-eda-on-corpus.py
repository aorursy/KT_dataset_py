import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline



list_children = pd.read_csv('../input/guide_to_files.csv')



list_children.loc[list_children['gender'] == 'M', 'gender'] = 'Male'

list_children.loc[list_children['gender'] == 'F', 'gender'] = 'Female'



list_children.head()
list_children['first_language'].value_counts().plot(kind='bar',figsize=(8,8), fontsize=15)
list_children['gender'].value_counts().plot(kind='pie',figsize=(8,8), fontsize=15)
dif_time = list_children['age_of_English_exposure_months'] - list_children['age_of_arrival_to_Canada_months']

print("Minimum: {}".format(dif_time.min()))

print("Maximum: {}".format(dif_time.max()))

print("Mean: {}".format(dif_time.mean()))



plt.title('Difference between arrival and exposure', fontsize=15)

plt.xlabel('# of Months', fontsize=15)

plt.ylabel('# of Children', fontsize=15)

dif_time.plot(kind = 'hist', figsize=(7,7), fontsize=15)
(list_children['age_at_recording_months'] - list_children['age_of_English_exposure_months']) == list_children['months_of_english']
list_children.tail(1)
import re



def getDialog(filename):

    dialog = []

    file = open(filename.strip(),'r')

    for line in file:

        if line[0] == '*':

            new_line = line[1:].strip()

            new_line = new_line[:4] + new_line[5:]

            dialog.append(new_line)

    file.close()

    return dialog



def analyzeDialog(dialog):

    pauses = 0

    ums = 0

    interrupts = 0

    for index, line in enumerate(dialog):

        if line[:3] == 'CHI':

            tmpline = line[4:]

            ums += len(re.findall(r"&-[^\s]*",tmpline)) 

            pauses += len(re.findall(r"\(\.\.?\)",tmpline))

            interrupts += len(re.findall(r"<[^>]*>",tmpline))

            # Commented code removes some imperfections in the speech.

            #while len(re.findall(r"&-[^\s]*",tmpline)) != 0:

            #    match = re.search(r"&-[^\s]*",tmpline)

            #    tmpline = tmpline[:match.start()] + tmpline[match.end():]

            #    ums += 1

            #while len(re.findall(r"\(\.\.?\)",tmpline)) != 0:

            #    match = re.search(r"\(\.\.?\)",tmpline)

            #    tmpline = tmpline[:match.start()] + tmpline[match.end():]

            #    pauses += 1

            #while len(re.findall(r"<[^>]*>",tmpline)) != 0:

            #    match = re.search(r"<[^>]*>",tmpline)

            #    tmpline = tmpline[:match.start()] + tmpline[match.end():]

            #    interrupts += 1

            #while len(re.findall(r"\[[^\]]*\]",tmpline)) != 0:

            #    match = re.search(r"\[[^\]]*\]",tmpline)

            #    tmpline = tmpline[:match.start()] + tmpline[match.end():]

            #print(tmpline)

            #dialog[index] = dialog[index][:4] + tmpline                 

    return (pauses, ums, interrupts)
list_children['pauses'] = 0

list_children['ums'] = 0

list_children['interrupts'] = 0

list_children['imperfections'] = 0

for index, row in list_children.iterrows():

    #print(row)



    dialog = getDialog('../input/{}'.format(row['file_name']))

    res = analyzeDialog(dialog)

    list_children.loc[index, 'pauses'] = res[0]

    list_children.loc[index, 'ums'] = res[1]

    list_children.loc[index, 'interrupts'] = res[2]

    list_children.loc[index, 'imperfections'] = sum(res)

list_children[['first_language', 'months_of_english', 'pauses', 'ums','interrupts', 'imperfections']].head(23)
print("Correlation between imperfections and months of English: {}".format(

    list_children[['imperfections','months_of_english']].corr())

)
print("Correlation between imperfections and age of child: {}".format(

    list_children[['imperfections','age_at_recording_months']].corr())

)
print("Correlation between ums and age of child: {}".format(

    list_children[['ums','age_at_recording_months']].corr())

)

list_children.plot(x='age_at_recording_months',y='ums', kind='scatter',figsize=(7,7),fontsize=15)

plt.title("Age and Ums", fontsize=15)

plt.xlabel("Age", fontsize=15)

plt.ylabel("Number of Ums", fontsize=15)

plt.show()
print("Correlation between pauses and age of child: {}".format(

    list_children[['pauses','age_at_recording_months']].corr())

)

list_children.plot(x='age_at_recording_months',y='pauses', kind='scatter',figsize=(7,7),fontsize=15)

plt.title("Age and Pauses", fontsize=15)

plt.xlabel("Age", fontsize=15)

plt.ylabel("Number of pauses", fontsize=15)

plt.show()
print("Correlation between pauses and age of child: {}".format(

    list_children[['pauses','months_of_english']].corr())

)

list_children.plot(x='months_of_english',y='pauses', kind='scatter',figsize=(7,7),fontsize=15)

plt.title("Months learning and pauses", fontsize=15)

plt.xlabel("Months learning Enlgish", fontsize=15)

plt.ylabel("Number of Pauses", fontsize=15)

plt.show()
print("Correlation between interruptions and age of child: {}".format(

    list_children[['interrupts','age_at_recording_months']].corr())

)

list_children.plot(x='age_at_recording_months',y='interrupts', kind='scatter',figsize=(7,7),fontsize=15)

plt.title("Age and interruptions", fontsize=15)

plt.xlabel("Age", fontsize=15)

plt.ylabel("Number of interruptions", fontsize=15)

plt.show()
plt.title("Language and ums", fontsize=15)

list_children.groupby(['first_language'])['ums'].sum().plot(kind = 'bar', figsize= (7,7), fontsize=15)

plt.xlabel("First Language", fontsize=15)

plt.ylabel("Number of ums", fontsize=15)

plt.show()



plt.title("Language and average number of ums", fontsize=15)

list_children.groupby(['first_language'])['ums'].mean().plot(kind = 'bar', figsize= (7,7), fontsize=15)

plt.xlabel("First Language", fontsize=15)

plt.ylabel("Average number of ums", fontsize=15)

plt.show()
plt.title("Language and pauses", fontsize=15)

list_children.groupby(['first_language'])['pauses'].sum().plot(kind = 'bar', figsize= (7,7), fontsize=15)

plt.xlabel("First Language", fontsize=15)

plt.ylabel("Number of pauses", fontsize=15)

plt.show()



plt.title("Language and average pauses", fontsize=15)

list_children.groupby(['first_language'])['pauses'].mean().plot(kind = 'bar', figsize= (7,7), fontsize=15)

plt.xlabel("First Language", fontsize=15)

plt.ylabel("Average number of pauses", fontsize=15)

plt.show()
plt.title("Language and total interruptions", fontsize=15)

list_children.groupby(['first_language'])['interrupts'].sum().plot(kind = 'bar', figsize= (7,7), fontsize=15)

plt.xlabel("First Language", fontsize=15)

plt.ylabel("Number of interrupts", fontsize=15)

plt.show()



plt.title("Language and average interruptions", fontsize=15)

list_children.groupby(['first_language'])['interrupts'].mean().plot(kind = 'bar', figsize= (7,7), fontsize=15)

plt.xlabel("First Language", fontsize=15)

plt.ylabel("Average number of interrupts", fontsize=15)

plt.show()
plt.title("Language and number of imperfections", fontsize=15)

list_children.groupby(['first_language'])['imperfections'].sum().plot(kind = 'bar', figsize= (7,7), fontsize=15)

plt.xlabel("First Language", fontsize=15)

plt.ylabel("Total number of imperfections", fontsize=15)

plt.show()



plt.title("Language and average amount of imperfections", fontsize=15)

list_children.groupby(['first_language'])['imperfections'].mean().plot(kind = 'bar', figsize= (7,7), fontsize=15)

plt.xlabel("First Language", fontsize=15)

plt.ylabel("Average number of imperfections", fontsize=15)

plt.show()