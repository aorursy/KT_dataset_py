from random import seed
from random import randint
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
type=['Normal','Fire','Water','Electric','Grass','Ice','Fighting','Poison','Ground','Flying','Psychic','Bug','Rock','Ghost','Dragon','Dark','Steel','Fairy']
#Data from Pokemon Database
Normal=[1,1,1,1,1,1,1,1,1,1,1,1,0.5,0,1,1,0.5,1]
Fire=[1,0.5,0.5,1,2,2,1,1,1,1,1,2,0.5,1,0.5,1,2,1]
Water=[1,2,0.5,1,0.5,1,1,1,2,1,1,1,2,1,0.5,1,1,1]
Electric=[1,1,2,0.5,0.5,1,1,1,0,2,1,1,1,1,0.5,1,1,1]
Grass=[1,0.5,2,1,0.5,1,1,0.5,2,0.5,1,0.5,2,1,0.5,1,0.5,1]
Ice=[1,0.5,0.5,1,2,0.5,1,1,2,2,1,1,1,1,2,1,0.5,1]
Fighting=[2,1,1,1,1,2,1,0.5,1,0.5,0.5,0.5,2,0,1,2,2,0.5]
Poison=[1,1,1,1,2,1,1,0.5,0.5,1,1,1,0.5,0.5,1,1,0,2]
Ground=[1,2,1,2,0.5,1,1,2,1,0,1,0.5,2,1,1,1,2,1]
Flying=[1,1,1,0.5,2,1,2,1,1,1,1,2,0.5,1,1,1,0.5,1]
Psychic=[1,1,1,1,1,1,2,2,1,1,0.5,1,1,1,1,0,0.5,1]
Bug=[1,0.5,1,1,2,1,0.5,0.5,1,0.5,2,1,1,0.5,1,2,0.5,0.5]
Rock=[1,2,1,1,1,2,0.5,1,0.5,2,1,2,1,1,1,1,0.5,1]
Ghost=[0,1,1,1,1,1,1,1,1,1,2,1,1,2,1,0.5,1,1]
Dragon=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,0.5,0]
Dark=[1,1,1,1,1,1,0.5,1,1,1,2,1,1,2,1,0.5,1,0.5]
Steel=[1,0.5,0.5,0.5,1,2,1,1,1,1,1,1,2,1,1,1,0.5,2]
Fairy=[1,0.5,1,1,1,1,2,0.5,1,1,1,1,1,1,2,2,0.5,1]
#Creating a pandas dataframe
types=[Normal,Fire,Water,Electric,Grass,Ice,Fighting,Poison,Ground,Flying,Psychic,Bug,Rock,Ghost,Dragon,Dark,Steel,Fairy]   
df = pd.DataFrame(columns=type)
for i in range(18):
    df.loc[i] = types[i]     
df.index = type
df
#Plotting a seaborn heatmap; Left to Right represent Attack, Top to Bottom represents Defense
fig, ax = plt.subplots(figsize=(18,18)) 
ax=sns.heatmap(df,annot=True,fmt='g',cmap=ListedColormap(['red', 'white', 'green']),linewidths=1)
ax.xaxis.tick_top()
plt.show()
#Function to separate all types into categories for a given type
def attack(type_num): 
    x2,x1,x_half,x0=[],[],[],[]
    
    for i,item in enumerate(types[type_num]):
        if item==2:
            x2.append(type[i])
        elif item==0.5:
            x_half.append(type[i])
        elif item==0:
            x0.append(type[i])
        else:
            x1.append(type[i])
 
    return x2,x1,x_half,x0
#Printing out the data obtained from attack function
def print_attack(type_num):
    x2,x1,x_half,x0=attack(type_num)
    print(f'{type[type_num]}: OFFENSE')
    print(f'Super Effective against: {x2}')
    print(f'Not Very Effective against: {x_half}')
    print(f'Normal Damage: {x1}')
    print(f'No Effect: {x0}')
#Function to separate all types into categories for a given type
def defense(type_num):
    x2,x1,x_half,x0=[],[],[],[]
    
    for index,i in enumerate(types):
        item=i[type_num]
        if item==2:
            x2.append(type[index])
        elif item==0.5:
            x_half.append(type[index])
        elif item==0:
            x0.append(type[index])
        else:
            x1.append(type[index])
    
    return x2,x1,x_half,x0
#Printing out the data obtained from defense function
def print_defense(type_num):
    x2,x1,x_half,x0=defense(type_num)
    print(f'{type[type_num]}: DEFENSE')
    print(f'Very Weak against: {x2}')
    print(f'Good Defence against: {x_half}')
    print(f'Normal Defence: {x1}')
    print(f'Resistant to: {x0}')
#Dataframe to store the number of types under each category
ranks = pd.DataFrame(columns=['Super Effective Against','Strong Defense Against','Immune To','No effect Against','Not very Effective Against','Weak Defence Against','Normal Offense','Normal Defense'])
for i in range(18):
    d2,d1,d_half,d0=defense(i)
    a2,a1,a_half,a0=attack(i)
    row=[len(a2),len(d_half),len(d0),len(a0),len(a_half),len(d2),len(a1),len(d1)]
    ranks.loc[i] = row   
ranks.index = type
ranks = ranks.astype(int)
ranks
#Seaborn heatmap for the above dataframe
fig, ax = plt.subplots(figsize=(20,9)) 
ax=sns.heatmap(ranks,annot=True,fmt='g',cmap='YlGnBu',linewidths=1)
ax.xaxis.tick_top()
plt.show()
fig, ax = plt.subplots(figsize=(6,6)) 
ax.bar(type,ranks["Super Effective Against"])
plt.title('Super Effective Against')
plt.xlabel('Types')
plt.ylabel('No of types')
plt.xticks(rotation=90)
plt.show()
#Plotting graphs for each type for each category
plt.figure(figsize=(18,12))
num_rows=3
num_cols=3
for i in range(8):
    plt.subplot(num_rows,num_cols,i+1)
    if i<3:
        plt.bar(type,ranks[ranks.columns[i]],color='green')
    elif i<6:
        plt.bar(type,ranks[ranks.columns[i]],color='red')   
    else:
        plt.bar(type,ranks[ranks.columns[i]],color='blue')
    plt.title(ranks.columns[i])
    plt.xlabel('Types')
    plt.ylabel('No of types')
    plt.xticks(rotation=90)
plt.tight_layout(h_pad=1.0)
plt.show()
#display dataframe with scores
scores= pd.DataFrame(columns=['Attack Score','Defense Score','Overall Score'])
for i in range(18):
    d2,d1,d_half,d0=defense(i)
    a2,a1,a_half,a0=attack(i)
    att_sc= 2*len(a2) + 1*len(a1) - 2*len(a_half) - 4*len(a0)
    def_sc= 2*len(d_half) + 1*len(d1) - 2*len(d2) + 4*len(d0)
    tot_sc=att_sc+def_sc
    row=[att_sc,def_sc,tot_sc]
    scores.loc[i] = row   
scores.index = type
scores
#Plotting graphs for each type based on scores
plt.figure(figsize=(18,6))
num_rows=1
num_cols=3
for i in range(3):
    plt.subplot(num_rows,num_cols,i+1)
    plt.bar(type,scores[scores.columns[i]])
    plt.title(scores.columns[i])
    plt.xlabel('Types')
    plt.ylabel('Score')
    plt.xticks(rotation=90)
plt.tight_layout(h_pad=1.0)
plt.show()
#Sorted according to overall score
sorted_scores=scores.sort_values(by=['Overall Score'],ascending=False)
sorted_scores
#Plotting each type according to overall score
fig, ax = plt.subplots(figsize=(9,6)) 
ax.bar(sorted_scores.index,sorted_scores['Overall Score'])
plt.title('Overall Score')
plt.xlabel('Types')
plt.ylabel('Score')
plt.xticks(rotation=90)
plt.show()
#Function to get different attack categories for a combination of any two types
def comb_attack(type_num1,type_num2): 
    t1_x2,t1_x1,t1_xhalf,t1_x0=attack(type_num1)
    t2_x2,t2_x1,t2_xhalf,t2_x0=attack(type_num2)
    x2,x1,x_half,x0=[],[],[],[]
    for t in type:
        if (t in t1_x2 or t in t2_x2):
            x2.append(t)
        elif(t in t1_x1 or t in t2_x1):
            x1.append(t)
        elif(t in t1_xhalf or t in t2_xhalf):
            x_half.append(t)
        elif(t in t1_x0 or t in t2_x0):
            x0.append(t)
            
    return x2,x1,x_half,x0           
#Printing out the above data
def print_comb_attack(type_num1,type_num2):
    x2,x1,x_half,x0=comb_attack(type_num1,type_num2)
    print(f'{type[type_num1]}-{type[type_num2]}: OFFENSE')
    print(f'Super Effective against: {x2}')
    print(f'Not Very Effective against: {x_half}')
    print(f'Normal Damage: {x1}')
    print(f'No Effect: {x0}')
#Function to get different defense categories for a combination of any two types
def comb_defense(type_num1,type_num2): 
    t1_x2,t1_x1,t1_xhalf,t1_x0=defense(type_num1)
    t2_x2,t2_x1,t2_xhalf,t2_x0=defense(type_num2)
    x4,x2,x1,x_half,x_quarter,x0=[],[],[],[],[],[]
    for t in type:
        if(t in t1_x0 or t in t2_x0):
            x0.append(t)
        elif (t in t1_x2 and t in t2_x2):
            x4.append(t)
        elif(t in t1_x2 and t in t2_x1):
            x2.append(t)
        elif(t in t1_x2 and t in t2_xhalf):
            x1.append(t)
        elif (t in t1_x1 and t in t2_x2):
            x2.append(t)
        elif(t in t1_x1 and t in t2_x1):
            x1.append(t)
        elif(t in t1_x1 and t in t2_xhalf):
            x_half.append(t)
        elif (t in t1_xhalf and t in t2_x2):
            x1.append(t)
        elif(t in t1_xhalf and t in t2_x1):
            x_half.append(t)
        elif(t in t1_xhalf and t in t2_xhalf):
            x_quarter.append(t)
            
    return x4,x2,x1,x_half,x_quarter,x0
#Printing out the above data
def print_comb_defense(type_num1,type_num2):
    x4,x2,x1,x_half,x_quarter,x0=comb_defense(type_num1,type_num2)
    print(f'{type[type_num1]}-{type[type_num2]}: DEFENSE')
    print(f'Very Weak against (x4): {x4}')
    print(f'Weak against (x2): {x2}')
    print(f'Very Good Defence against (x1/4): {x_quarter}')
    print(f'Good Defence against(x1/2): {x_half}')
    print(f'Normal Defence: {x1}')
    print(f'Resistant to: {x0}')
#Plotting the number of types under each category for a combination of two given types
def plot_comb(type_num1,type_num2):
    a2,a1,a_half,a0=comb_attack(type_num1,type_num2)
    d4,d2,d1,d_half,d_quarter,d0=comb_defense(type_num1,type_num2)
    cat=['Super Effective Against','Strong Defense(x1/4) against','Good Defense(x1/2) Against','Immune To','No effect on','Not very effective Against','Very Weak(x4) Against','Weak(x2) Against','Normal Offense','Normal Defense']
    num_cat=[len(a2),len(d_quarter),len(d_half),len(d0),len(a0),len(a_half),len(d4),len(d2),len(a1),len(d1)]
    fig, ax = plt.subplots(figsize=(6,6)) 
    ax.bar(cat,num_cat,color=['green','green','green','green','red','red','red','red','blue','blue'])
    plt.title(f'{type[type_num1]}-{type[type_num2]}')
    plt.ylabel('No of types')
    plt.xticks(rotation=90)
    plt.show()
#Randomly generates two numbers which correspond to types
def gen_types():
    t1=t2=0
    while (t1==t2):
        t1=randint(0,17)
        t2=randint(0,17)
    print(type[t1]+"-"+type[t2])
    return t1,t2
t1,t2=gen_types()
print_attack(t1)
print_defense(t1)
print_attack(t2)
print_defense(t2)
type[t1],type[t2]
print_comb_attack(t1,t2)
print_comb_defense(t1,t2)
plot_comb(t1,t2)
#Dataframe to store scores of combined types
comb_scores= pd.DataFrame(columns=['Attack Score','Defense Score','Overall Score'])
c=0
comb_types=[]
for i in range(18):
    for j in range(i+1,18):
        d4,d2,d1,d_half,d_quarter,d0=comb_defense(i,j)
        a2,a1,a_half,a0=comb_attack(i,j)
        att_sc= 2*len(a2) + 1*len(a1) - 2*len(a_half) - 4*len(a0)
        def_sc= 4*len(d_quarter) + 2*len(d_half) + 1*len(d1) - 2*len(d2) - 4*len(d4) + 4*len(d0)
        tot_sc=att_sc+def_sc
        row=[att_sc,def_sc,tot_sc]
        comb_scores.loc[c] = row 
        c+=1
        comb_types.append(type[i]+"-"+type[j])
comb_scores.index = comb_types
comb_scores
#Sorting them according to overall scores
sorted_comb_scores=comb_scores.sort_values(by=['Overall Score'],ascending=False)
sorted_comb_scores
print(sorted_comb_scores.to_string())
plt.hist(sorted_comb_scores["Overall Score"],bins=10)
plt.ylabel('Number of Types')
plt.xlabel('Scores');
