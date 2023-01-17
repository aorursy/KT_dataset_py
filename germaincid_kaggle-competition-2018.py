# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Any results you write to the current directory are saved as output.
schema = pd.read_csv("../input/SurveySchema.csv")
mult = pd.read_csv("../input/multipleChoiceResponses.csv",dtype=np.object)
free = pd.read_csv("../input/freeFormResponses.csv",dtype=np.object)
multmas = mult[mult.Q4 == 'Master’s degree']
multdoc = mult[mult.Q4 == 'Doctoral degree']
#Which best describes your undergraduate mejor? (Master´s degree)

print('\n','\n','Which best describes your undergraduate mejor? (Master´s degree)','\n')

ansm5 = ((multmas.Q5.value_counts()).keys()).tolist()
ansm5 = ansm5[:(len(ansm5))]

fasm5 = ((multmas.Q5.dropna()).value_counts()).tolist()
fasm5 = fasm5[:len(fasm5)]

for i in range(len(fasm5)):
    print(ansm5[i],':',fasm5[i])

#Which best describes your undergraduate mejor? (Doctoral degree)
print('\n','\n','Which best describes your undergraduate mejor? (Doctoral degree)','\n')

ansd5 = ((multdoc.Q5.value_counts()).keys()).tolist()
ansd5 = ansd5[:(len(ansd5))]

fasd5 = ((multdoc.Q5.dropna()).value_counts()).tolist()
fasd5 = fasd5[:len(fasd5)]

for i in range(len(fasd5)):
    print(ansd5[i],':',fasd5[i])
    
##############PLOTS BARS###########################

#Set axes of the figure 1
names_m = ansm5
values_m = fasm5
#Set axes of the figure2
names_d = ansd5
values_d = fasd5

fig, axs = plt.subplots(1, 2, figsize=(18, 9))
axs[0].bar(names_m,values_m,color='b')
axs[0].set_title("Master´s degree")
plt.grid(True)

axs[1].bar(names_d,values_d,color='r')
axs[1].set_title("Doctoral degree")
fig.suptitle('Best description of undergraduate major')

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)
    plt.grid(True)
#Select the title most similar to your current role (or most recentitleir retired)? (Master´s degree)
print('\n','\n','Select the title most similar to your current role (or most recentitleir retired)? (Master´s degree)','\n')
ansm6 = ((multmas.Q6.value_counts()).keys()).tolist()
ansm6 = ansm6[:(len(ansm6))]

fasm6 = ((multmas.Q6.dropna()).value_counts()).tolist()
fasm6 = fasm6[:len(fasm6)]

for i in range(len(fasm6)):
    print(ansm6[i],':',fasm6[i])
print('\n','\n','Select the title most similar to your current role (or most recentitleir retired)? (Doctoral degree)','\n')
    
#Select the title most similar to your current role? (Doctoral degree)
ansd6 = ((multdoc.Q6.value_counts()).keys()).tolist()
ansd6 = ansd6[:(len(ansd6)-1)]

fasd6 = ((multdoc.Q6.dropna()).value_counts()).tolist()
fasd6 = fasd6[:len(fasd6)-1]

for i in range(len(fasd6)):
    print(ansd6[i],':',fasd6[i])

############################# PLOT BAR #########################################
###################PLOT BARS############################
#Set axes of the figure 1
names_m = ansm6
values_m = fasm6
#Set axes of the figure 2
names_d = ansd6
values_d = fasd6


fig, axs = plt.subplots(1, 2, figsize=(18, 9))
axs[0].bar(names_m,values_m,color='b')
axs[0].set_title("Maester´s degree")
#plt.xticks(rotation='vertical')
#plt.legend()
plt.grid(True)

axs[1].bar(names_d,values_d,color='r')
axs[1].set_title("Doctoral degree")
#plt.xticks(rotation='vertical')
fig.suptitle('Title most similar for current role (or most recentitleir retired)')

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)
    plt.grid(True)
#In what industry is your current employer/contract (or your most recent employer)? (Master´s degree)
print('\n','\n','In what industry is your current employer/contract (or your most recent employer)? (Master´s degree)','\n')
ansm7 = ((multmas.Q7.value_counts()).keys()).tolist()
ansm7 = ansm7[:(len(ansm7))]

fasm7 = ((multmas.Q7.dropna()).value_counts()).tolist()
fasm7 = fasm7[:len(fasm7)]

for i in range(len(fasm7)):
    print(ansm7[i],':',fasm7[i])
print('\n','\n','In what industry is your current employer/contract (or your most recent employer)? (Doctoral degree)','\n')
    
#In what industry is your current employer/contract (or your most recent employer)? (Doctoral degree)
ansd7 = ((multdoc.Q7.value_counts()).keys()).tolist()
ansd7 = ansd7[:(len(ansd7)-1)]

fasd7 = ((multdoc.Q7.dropna()).value_counts()).tolist()
fasd7 = fasd7[:len(fasd7)-1]

for i in range(len(fasd7)):
    print(ansd7[i],':',fasd7[i])

###############################################
#Set axes of the figure 1
names_m = ansm7
values_m = fasm7
#Set axes of the figure 2
names_d = ansd7
values_d = fasd7

fig, axs = plt.subplots(1, 2, figsize=(18, 9))
axs[0].bar(names_m,values_m,color='b')
axs[0].set_title("Masters degree")

axs[1].bar(names_d,values_d,color='r')
axs[1].set_title("Doctoral degree")
fig.suptitle('Industry of your current employer/contract')

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)
    plt.grid(True)
#What is your current yearly compresation (aproximate $USD)? (Master´s degree)
print('\n','\n','What is your current yearly compresation (aproximate $USD)? (Master´s degree)','\n')
ansm9 = ((multmas.Q9.value_counts()).keys()).tolist()
ansm9 = ansm9[:(len(ansm9))]

fasm9 = ((multmas.Q9.dropna()).value_counts()).tolist()
fasm9 = fasm9[:len(fasm9)]

for i in range(len(fasm9)):
    print(ansm9[i],':',fasm9[i])
print('\n','\n','What is your current yearly compresation (aproximate $USD)? (Doctoral degree)','\n')
    
#What is your current yearly compresation (aproximate $USD)? (Doctoral degree)
ansd9 = ((multdoc.Q9.value_counts()).keys()).tolist()
ansd9 = ansd9[:(len(ansd9)-1)]

fasd9 = ((multdoc.Q9.dropna()).value_counts()).tolist()
fasd9 = fasd9[:len(fasd9)-1]

for i in range(len(fasd9)):
    print(ansd9[i],':',fasd9[i])
################# PLOT ####################
#Set axes of the subplot 1
names_m = ansm9
values_m = fasm9
#Set axes of the subplt 2
names_d = ansd9
values_d = fasd9

fig, axs = plt.subplots(1, 2, figsize=(18, 9))
axs[0].bar(names_m,values_m,color='b')
axs[0].set_title("Master´s degree")

axs[1].bar(names_d,values_d,color='r')
axs[1].set_title("Doctoral degree")
fig.suptitle('Compresation (aproximate $USD)')

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)
    plt.grid(True)
#Which specific data visualization library or tool have you used the most? (Master´s degree)
print('\n','\n','Which specific data visualization library or tool have you used the most? (Master´s degree)','\n')
ansm22 = ((multmas.Q22.value_counts()).keys()).tolist()
ansm22 = ansm22[:(len(ansm22))]

fasm22 = ((multmas.Q22.dropna()).value_counts()).tolist()
fasm22 = fasm22[:len(fasm22)]

for i in range(len(fasm22)):
    print(ansm22[i],':',fasm22[i])
print('\n','\n','Which specific data visualization library or tool have you used most? (Doctoral degree)','\n')
    
#What is your current yearly compresation (aproximate $USD)? (Doctoral degree)
ansd22 = ((multdoc.Q22.value_counts()).keys()).tolist()
ansd22 = ansd22[:(len(ansd22)-1)]

fasd22 = ((multdoc.Q22.dropna()).value_counts()).tolist()
fasd22 = fasd22[:len(fasd22)-1]

for i in range(len(fasd22)):
    print(ansd22[i],':',fasd22[i])
################# PLOT ####################
#Set axes of the subplot 1
names_m = ansm22
values_m = fasm22
#Set axes of the subplot 2
names_d = ansd22
values_d = fasd22

fig, axs = plt.subplots(1, 2, figsize=(18, 9))
axs[0].bar(names_m,values_m,color='b')
axs[0].set_title("Master´s degree")
plt.grid(True)

axs[1].bar(names_d,values_d,color='r')
axs[1].set_title("Doctoral degree")
fig.suptitle('Specific data visualization library or tool')

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)
    plt.grid(True)
#What is the type of data that you currently interact with most often at work or school? (Master´s degree)
print('\n','\n','What is the type of data that you currently interact with most often at work or school? (Master´s degree)','\n')
ansm32 = ((multmas.Q32.value_counts()).keys()).tolist()
ansm32 = ansm32[:(len(ansm32))]

fasm32 = ((multmas.Q32.dropna()).value_counts()).tolist()
fasm32 = fasm32[:len(fasm32)]

for i in range(len(fasm32)):
    print(ansm32[i],':',fasm32[i])
print('\n','\n','What is the type of data that you currently interact with most often at work or school? (Doctoral degree)','\n')
    
#What is the type of data that you currently interact with most often at work or school? (Doctoral degree)
ansd32 = ((multdoc.Q32.value_counts()).keys()).tolist()
ansd32 = ansd32[:(len(ansd32)-1)]

fasd32 = ((multdoc.Q32.dropna()).value_counts()).tolist()
fasd32 = fasd32[:len(fasd32)-1]

for i in range(len(fasd32)):
    print(ansd32[i],':',fasd32[i])

#######################PLOT WITH BARS#########################
#Set axes of the subplot 1
names_m = ansm32
values_m = fasm32
#Set axes of the subplot 2
names_d = ansd32
values_d = fasd32

fig, axs = plt.subplots(1, 2, figsize=(18, 9))
axs[0].bar(names_m,values_m,color='b')
axs[0].set_title("Maester´s degree")
plt.grid(True)

axs[1].bar(names_d,values_d,color='r')
axs[1].set_title("Doctoral degree")
fig.suptitle('Data what currently interact with most often')

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)
    plt.grid(True)