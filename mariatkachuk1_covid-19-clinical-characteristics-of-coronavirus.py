#import libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches



#type 1 plot

#create dataframe taking the data in the research

age1=pd.DataFrame({'Age':['0-14','15-49','50-64','>64','0-14','15-49','50-64','>64','0-14','15-49','50-64','>64',],

                  'Cases':['Nonsevere','Nonsevere','Nonsevere','Nonsevere','Severe','Severe','Severe','Severe','Lethal','Lethal','Lethal','Lethal'],

                 'Numbers':[8,490,241,109,1,67,51,44,0,12,21,32]

                })

#set plot background and remove chart edges

sns.set_style('whitegrid')



sns.set_style( {'axes.spines.left': False,

    'axes.spines.bottom': False,

    'axes.spines.right': False,

    'axes.spines.top': False})



#create list of numbers from the research for further plotting

num=[9,557,292,153,9,557,292,153,9,557,292,153,9,557,292,153] 



#set plot size and create a barplot

fig,ax=plt.subplots(figsize=(16,10))

sns.barplot(age1['Age'],age1['Numbers'],ax=ax,hue=age1['Cases'],palette='GnBu')



#add annotations

count=0

for p in ax.patches:

    if count==1:

        ax.annotate(format(p.get_height()/num[count]*100, '.2f')+'%',(p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, -6), textcoords = 'offset points',size=9)

    elif count!=1:

        ax.annotate(format(p.get_height()/num[count]*100, '.2f')+'%',(p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 6), textcoords = 'offset points',size=9)

    print(p.get_height(),' ',num[count])

    count+=1

ax.set_ylabel('Ammount of Patients')

ax.set_xlabel('Age')

ax.set_title('Patients by Age')

plt.legend(bbox_to_anchor=(1.05, 0.98))

plt.show()



#chart2

#create data lists of info from the research

labels=['0-14','15-49','50-64','>64']

sizes=[0.6,41.1,31.3,27]

#color list to fill for different age groups

colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']



fig1, ax1 = plt.subplots(figsize=(16,10))

theme = plt.get_cmap('PuBuGn')

ax1.set_prop_cycle("color", [theme(1. * i / len(sizes))

                             for i in range(len(sizes))])

 

ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=45)

ax1.axis('equal')  

plt.title('Severe Patients by Age',fontsize=16)

plt.legend(title='Age',labels=labels,)



plt.show()



#pie chart 3



labels=['15-49','50-64','>64']

sublabels=['17.91%','82.09%','41.18%','58.82%','72.73%','27.27%']

c=plt.cm.PuBuGn                            



sizes=[41.1,31.3,27.6]

sizes_inner=[7.36,33.74,12.89,18.41,19.64,7.96]



fig,ax=plt.subplots(figsize=(16,10))

ax.axis('equal')



mypie,_=ax.pie(sizes,radius=1.3,colors=[c(0.1),c(0.5),c(0.9)],startangle=55,labels=labels)

plt.setp(mypie, width=0.3,edgecolor='white')

plt.subplots_adjust(top=0.9)

plt.title('Severe Patients by Age\n lethal cases by age group\n                   \n                   ',fontsize=15)



mypie2,_=ax.pie(sizes_inner, radius=1.3-0.3,startangle=55,labels=sublabels,labeldistance=0.7,

                colors=[c(0.7),c(0.3),c(0.7),c(0.3),c(0.7),c(0.3)])

plt.setp(mypie2,width=0.4,edgecolor='white')

plt.margins(0,0)

dark_patch = mpatches.Patch(color=c(0.7),label='Died')

dark1_patch = mpatches.Patch(color=c(0.3),label='Recovered')

plt.legend(handles=[dark_patch,dark1_patch])

plt.show()

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches



sns.set_style( {'axes.spines.left': False,

    'axes.spines.bottom': False,

    'axes.spines.right': False,

    'axes.spines.top': False})

gender=pd.DataFrame({'Gender':['0','1'],'Count':[459,637]})



fig,ax=plt.subplots(figsize=(15,10))

plt.scatter([2,3],[4,4],s=gender['Count']*35,alpha=0.7,edgecolors='black',linewidth=2,c=['#8e9609','#6d8de8'])

plt.xlim(1,4)



plt.annotate('41.9%',(1.95,4))

plt.annotate('58.1%',(2.95,4))

plt.annotate('Women',(1.9,4.08),fontsize=14)

plt.annotate('Men',(2.9,4.08),fontsize=14)

plt.xticks([])

plt.yticks([])

plt.title('Disease in Men vs Women',fontsize=16)

#chart 5

sns.set_style('white')

sns.set_style( {'axes.spines.left': False,

    'axes.spines.bottom': True,

    'axes.spines.right': False,

    'axes.spines.top': False})

g=pd.DataFrame({'f':[386,73,22,437],'m':[537,100,45,592]})

labelsf=['41.8%','42.2%','32.2%','42.5%']

labelsm=['58.2%','57.8%','67.2%','57.5%']

yticksl=['Nonsevere','Severe','Died','Recovered']





fix, ax = plt.subplots(1,2,figsize=(15,10),gridspec_kw={ 'wspace': 0.4})

ax[0].set_xlim(0,700)

ax[0].invert_xaxis()

ax[0].barh([4,3,2,1],g.f,color=sns.color_palette('bone'))

ax[0].set_yticklabels(' ')

plt.yticks([])

ax[0].title.set_text('Women')

num=[926,173,67,1032]

count=0

for p in ax[0].patches:

    ax[0].annotate(format(p.get_width()/num[count]*100, '.2f')+'%',(p.get_width(),p.get_y() + p.get_height() / 2.), ha = 'center', va = 'center', xytext = (-25, -5), textcoords = 'offset points',size=10)

    count+=1







ax[1].barh([4,3,2,1],g.m,color=sns.color_palette('pink'))

ax[1].set_xlim(0,700)

plt.yticks([4,3,2,1])



ax[1].set_yticklabels(['Nonsevere      ','Severe        ','Died         ','Recovered    '],fontsize=12)

ax[1].title.set_text('Men')

num=[926,173,67,1032]

count=0

for p in ax[1].patches:

    ax[1].annotate(format(p.get_width()/num[count]*100, '.2f')+'%',(p.get_width(),p.get_y() + p.get_height() / 2.), ha = 'center', va = 'center', xytext = (25, -5), textcoords = 'offset points',size=10)

    count+=1

#chart 6

df=pd.DataFrame({'Status':['86.9% - Never smoked\n nonsevere','1.3%\n Former\n smoker\n non\nsevere','11.8% - Current\n smoker\n nonsevere',

                           '77.9% - Never smoked\n severe','5.2%\nFormer\n smoker\n severe','16.9% - Current\n smoker\n severe',

                           '66.7% - Never smoked\n lethal','7.6%\nFormer\n smoker\n lethal','25.8% - Current\n smoker\n lethal',

                           '86.7% - Never smoked\n recovered','1.6% - Former\n smoker\n recovered','11.8% - Current\n smoker\n recovered'],

                 'Numbers':[793,12,108,134,9,29,44,5,17,883,16,120]



                 })



import squarify

fig,ax=plt.subplots()

fig.set_size_inches((25,15))   

plt.rcParams.update({'font.size': 14}) 

colors=['#632504','#a84b19','#db7f4d','#856e08','#d4b737','#f5dd76','#088239','#40c776','#aef5ca','#acdde3','#1c8f9e','#8ae2ed']

squarify.plot(df['Numbers'],label=df['Status'],color=sns.color_palette('BrBG')) 

p1=mpatches.Patch(color=sns.color_palette('BrBG')[0],label='86.9% Never smoked')

p2=mpatches.Patch(color=sns.color_palette('BrBG')[1],label='1.3% Former smoker')

p3=mpatches.Patch(color=sns.color_palette('BrBG')[2],label='11.8% Current smoker')



p4=mpatches.Patch(color=sns.color_palette('BrBG')[3],label='77.9% - Never smoked')

p5=mpatches.Patch(color=sns.color_palette('BrBG')[4],label='5.2% Former smoker')

p6=mpatches.Patch(color=sns.color_palette('BrBG')[5],label='16.9% Current smoker')

l1=plt.legend(handles=[p4,p5,p6],bbox_to_anchor=(1.,0.4),title='Among Severe 100%',fontsize=15)





p7=mpatches.Patch(color=sns.color_palette('BrBG')[0],label='66.7% - Never smoked')

p8=mpatches.Patch(color=sns.color_palette('BrBG')[1],label='7.6% Former smoker')

p9=mpatches.Patch(color=sns.color_palette('BrBG')[2],label='25.8 Current smoker')

l2=plt.legend(handles=[p7,p8,p9],bbox_to_anchor=(1.,0.2),title='Among Died 100%',fontsize=15)



p10=mpatches.Patch(color=sns.color_palette('BrBG')[3],label='86.7% - Never smoked')

p11=mpatches.Patch(color=sns.color_palette('BrBG')[4],label='1.6% Former smoker')

p12=mpatches.Patch(color=sns.color_palette('BrBG')[5],label='11.8% Current smoker')

l3=plt.legend(handles=[p10,p11,p12],bbox_to_anchor=(1.,0.7),title='Among Recovered 100%',fontsize=15)







plt.legend(handles=[p1,p2,p3],bbox_to_anchor=(1.,1.),title='Among Nonsevere 100%' ,fontsize=15) 

plt.xticks([])

plt.yticks([])   

plt.title('Influence of Smoking on Disease Outcome',fontsize=20)    

plt.gca().add_artist(l1)

plt.gca().add_artist(l2)

plt.gca().add_artist(l3)

plt.show()

    
    

#chart 7



sns.set_style('white')

sns.set_style( {'axes.spines.left': False,

    'axes.spines.bottom': False,

    'axes.spines.right': False,

    'axes.spines.top': False})   

   

fix, ax=plt.subplots(figsize=(20,10))



#create dataframe with the necessary data from the research

"""I decided to focus on the visual side of this plot and created x and y coordinates on my own"""



df=pd.DataFrame({'names':['Chronic obstructive pulmonary disease','Diabetes','Hypertension','Coronary heart disease',

       'Cerebrovascular disease','Hepatitis B','Cancer','Chronic renal disease','Immunodeficiency'],

'numbers':[1.1,7.4,15.0,2.5,1.4,2.1,0.9,0.7,0.2],

'aa':[1200,8100,16500, 2700, 1500, 2300,1000,800,200],

'x':[ 5.1, 2.4,   4.8, 4,   3.2,    2.6,   1.65, 3.1, 5.55],

'y':[ 3.8, 5.5,   8.2, 3.6, 11.2,   9.2,   8.3,  2.7, 4.7] 

}) 

colors=['#8e9609','#6d8de8', '#71b5c9',  '#71c9a1', '#0f732d','#8a5f94', '#871b56', '#87651b', '#873a1b']



#create bubble charts with percentage distribution

plt.scatter(df.x,df.y,s=df.aa*3,color=colors,linewidth=2,alpha=0.7)

plt.barh(18,width=13,color='#73ebe1',alpha=0.6)

plt.barh(18,width=3.1,color='#478f89',alpha=0.6)



#add horisontal bars

plt.barh(17,width=1.8,color='#2c393b')

plt.barh(17,width=0.3,left=3.1,color='#2c393b')



#add annotations

ax.text(1.9, 16.8,'58.2%',fontsize=10)

ax.text(3.5,16.8,'3%',fontsize=10)

ax.text(1.5,17.8,'23.7%',fontsize=10)

ax.text(8,17.8,'76.3%',fontsize=10 )



#create legend

no=mpatches.Patch(color='#73ebe1',label='Coexisting disorder present')

yes=mpatches.Patch(color='#478f89',label='Coexisting disorder absent')

dead=mpatches.Patch(color='#2c393b',label='Died')

legend1=plt.legend(handles=[no,yes,dead],loc='upper right')



for a,b in enumerate(df.numbers):

    ax.text(df.x[a]-0.2,df.y[a]-0.2,str(b)+'%',fontsize=8)

 

one_patch = mpatches.Patch(color='#8e9609',label='Chronic obstructive pulmonary disease')

two_patch = mpatches.Patch( color='#6d8de8',label='Diabetes')

three_patch = mpatches.Patch(color='#71b5c9',label='Hypertension')

four_patch = mpatches.Patch( color='#71c9a1',label='Coronary heart disease')

five_patch = mpatches.Patch(color='#0f732d',label='Cerebrovascular disease')

six_patch = mpatches.Patch( color='#8a5f94',label='Hepatitis B')

seven_patch = mpatches.Patch(color='#871b56',label='Cancer')

eight_patch = mpatches.Patch( color='#87651b',label='Chronic renal disease')

nine_patch = mpatches.Patch(color='#873a1b',label='Immunodeficiency')



# create coordinates for the grey line depicting dead in the disorder group

x=[4.7,5.5,1.55,3.2,3.75,5.8,3.55,4.5,2.85,3.55,2.35,2.8,1.45,1.85,3,3.2]

y=[3.8,3.8,4,   4,  5.6, 5.6,2.7,2.7,10.6,10.6,8.1,8.1,7.6,7.6,2.2,2.2]



n=2

x_cut = [x[i * n:(i + 1) * n] for i in range((len(x) + n - 1) // n )]  

y_cut=[y[i * n:(i + 1) * n] for i in range((len(y) + n - 1) // n )]



for a in range(0,len(x_cut)-1):

    line1,=plt.plot(x_cut[a],y_cut[a],color='grey',linewidth=1,label='% of dead below the line')    

    



#coordinates for bubble plot in grey barplot

x=[10,10,10,10,10,10,10,10]

y=[1.5,3,5,6.7,8,8.7,9.5,10.5]

s=[700,1800,2400,600,400,100,100,200]

colors=['#8e9609','#6d8de8', '#71b5c9',  '#71c9a1', '#0f732d','#8a5f94', '#871b56', '#87651b']



#charts themselves

plt.scatter(x,y,s=s,color=colors,linewidth=2,alpha=0.7)

plt.bar(10,height=11,color='#2c393b',alpha=0.3,width=1.5,bottom=0.6)

text=['10.4%','26.9%','35.8%','9%','6%','1.5%','1.5%','3%']

for a in range(0,len(x)):

    ax.text(x[a]+0.8,y[a]-0.2,text[a],fontsize=10)



labels=['58%','22%','14%','22%','27%','4%','10%','10%']

x=[5.5, 1.5, 5.9, 4.4, 3.5,  1.8,  1,   2.7]

y=[3 ,  3,   5.1, 2.1, 12.1, 10.1, 7.2, 1.6]

for a in range(0,len(x)):

    ax.text(x[a],y[a],labels[a],fontsize=8)



#create legend for line

legend2=plt.legend(handles=[line1],loc='lower left')



ax.text(1,14,'Coexisting disorders and percentage of dead \n              per disorder',fontsize=16)

ax.text(8,14.5,'Coexising disorders among dead',fontsize=16)



plt.xlim(0,20)

plt.ylim(-1,21)

plt.xticks([])

plt.yticks([])

plt.title('Coexisitng Disorders',fontsize=16) 

plt.legend(handles=[one_patch,two_patch,three_patch,four_patch,five_patch,six_patch,seven_patch,eight_patch,nine_patch],loc='lower right')

plt.gca().add_artist(legend1)

plt.gca().add_artist(legend2)

plt.show()



#chart 8









import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches







sns.set_style('white')

sns.set_style( {'axes.spines.left': True,

    'axes.spines.bottom': True,

    'axes.spines.right': True,

    'axes.spines.top': True})



df=pd.DataFrame({'x':[2,2,2,2,2,2,2,2,2,2,2],

                'y':[637,393,31,204,454,25,56,15,19,144,55],

                'treatment':[ 'Intravenous antibiotics 58%',

                     'Oseltamivir 35.8%',

                     'Antifungal medication 2.8%',

                     'Systemic glucocorticoids 18.6%',

                     'Oxygen therapy 41.3%',

                     'Invasive mechanical ventilation 2.3%',

                     'Noninvasive mechanical ventilation 5.1%',

                     'Extracorporeal membrane oxygenation 0.5%',

                     'Continuous renal-replacement therapy 0.8%',

                     'Intravenous immune globulin 13.1%', 

                     'Admission to intensive care unit 5%']}

                )



#ax, fig=plt.subplots(figsize=(10,5))

g = sns.FacetGrid(df, row='treatment',height=1, 

                  aspect=7)

g.map(plt.bar,'x','y',color='pink');

g.set_titles("{row_name}",fontsize=7) 

g.despine(left=True,bottom=True)

g.set_xlabels('')

g.set_ylabels('')

g.set_xticklabels('')

g.set_yticklabels('')

g.fig.subplots_adjust(hspace=0.5)

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Treatment by Type in Percentages')


