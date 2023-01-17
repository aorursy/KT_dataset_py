import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os



import warnings

warnings.filterwarnings('ignore')  # 不发出警告

import time #  导入时间模块,记录运行时间
# 设定初始参数：游戏玩家100人，起始资金100元

person_n = [x for x in range(1,101)]

fortune = pd.DataFrame([100 for i in range(100)], index = person_n,columns=['r0'])

fortune.index.name = 'id'



round_r1 = pd.DataFrame({'pre_round':fortune['r0'],'lost':0})

choice_r1 = pd.Series(np.random.choice(person_n,100))     #每个人选一个id,把钱给他

gain_r1 = pd.DataFrame({'gain':choice_r1.value_counts()}) #汇总这一轮每个人的盈利情况



round_r1=round_r1.join(gain_r1)

round_r1.fillna(0,inplace=True) #没有拿到钱的，gain值为0

round_r1['r1']=round_r1['pre_round']-round_r1['lost']+round_r1['gain']

fortune=fortune.join(round_r1['r1'])

#fortune.head()
n=100

def game(data,roundi):

    if len(data[data[roundi - 1] ==0]) > 0:    # 当数据包含财富值为0的玩家时

        round_i=pd.DataFrame({'pre_round': data[roundi-1],'lost':0})

        round_i['lost'][round_i['pre_round']>0]=1  # 考虑情况：当某人的财富值降到0元时，

                                                     #他在该轮无需拿出1元钱给别人，但仍然有机会得到别人给出的钱

        round_i_players=round_i[round_i['pre_round']>0]

        choice_i=pd.Series(np.random.choice(person_n,len(round_i_players))) 

        gain_i=pd.DataFrame({'gain':choice_i.value_counts()})

        round_i=round_i.join(gain_i)

        round_i.fillna(0,inplace=True)

        return round_i['pre_round'] -  round_i['lost'] + round_i['gain']    

    else: # 当数据不包含财富值为0的玩家时

        round_i = pd.DataFrame({'pre_round':data[roundi-1],'lost':1}) # 设定每轮分配财富之前的情况

        choice_i = pd.Series(np.random.choice(person_n,100))

        gain_i = pd.DataFrame({'gain':choice_i.value_counts()})       # 这一轮中每个人随机指定给“谁”1元钱，并汇总这一轮每个人的盈利情况

        round_i = round_i.join(gain_i)

        round_i.fillna(0,inplace = True)

        return round_i['pre_round'] -  round_i['lost'] + round_i['gain']

        



# 设定初始参数：游戏玩家100人，起始资金100元

person_n = [x for x in range(1,101)]

fortune = pd.DataFrame([100 for i in range(100)], index = person_n)

fortune.index.name = 'id'



# 运行模型，模拟财富分配  

starttime = time.time() 



for  i  in range(1,10001):

    fortune[i]=game(fortune,i)

    

endtime = time.time() 

print('模型总共用时%i秒' % (endtime - starttime))



#绘图

fortuneT=fortune.T

os.getcwd()

'/kaggle/working'

def graph1(data,start,end,length):

    for n in list(range(start,end,length)):

        datai = data.iloc[n].sort_values().reset_index()[n]

        plt.figure(figsize = (5,3))

        plt.bar(datai.index,datai.values,color='gray',alpha = 0.8,width = 0.9)

        plt.ylim((0,380))

        plt.xlim((-10,110))

        plt.title('Round %d' % n)

        plt.xlabel('PlayerID')

        plt.ylabel('Fortune')

        plt.grid(color='gray', linestyle='--', linewidth=0.5)

        #plt.savefig('graph1_round_%d.png' % n, dpi=200)

graph1(fortuneT,0,10000,1000)
#some analysis

lastround=pd.DataFrame({'money':fortuneT.iloc[-1]}).sort_values('money',ascending = False).reset_index()

lastround['money_percent']=lastround['money']/lastround['money'].sum()

lastround['money_cum']=lastround['money_percent'].cumsum()



#pareto chart- wealth distribution

df=lastround

fig, ax = plt.subplots()

ax.bar(df.index, df["money"], color="C0")

ax2 = ax.twinx()

ax2.plot(df.index, df["money_cum"], color="C1", marker=".", ms=7)

ax.grid(alpha=0.2)

a=df.iloc[20]["money_cum"]

ax2.text(20,a,s='(20,%.1f%%)'%(a*100),weight ='bold',fontsize=15)

ax.set_title('Pareto chart ')

print('在这次模拟中，20%%的人掌握了%.1f%%的财富'%(a*100))



stats=fortuneT.iloc[-1].describe()

print('最后一轮中，最富的人有%i元,'%stats['max'],"相比于初始财富，翻了%.1f倍"%(stats['max']/100))



a=len(df[df['money']<100])/100 #财富少于100的比例

print('%.1f%%的人财富缩水'%(a*100))



#构建函数模型

def game2(data, roundi):

    round_i = pd.DataFrame({'pre_round':data[roundi-1],'lost':1}) # 设定每轮分配财富之前的情况

    choice_i = pd.Series(np.random.choice(person_n,100))

    gain_i = pd.DataFrame({'gain':choice_i.value_counts()})       # 这一轮中每个人随机指定给“谁”1元钱，并汇总这一轮每个人的盈利情况

    round_i = round_i.join(gain_i)

    round_i.fillna(0,inplace = True)

    return round_i['pre_round'] -  round_i['lost'] + round_i['gain'] # 合并数据，得到这一轮财富分配的结果



# 设定初始参数：游戏玩家100人，起始资金100元

person_n = [x for x in range(1,101)]

fortune = pd.DataFrame([100 for i in range(100)], index = person_n)

fortune.index.name = 'id'



#迭代1w次，模拟财富分配 

starttime = time.time()                     # 模型开始时间

for round in range(1,10001):

    fortune[round] = game2(fortune,round)   # 进行17000轮随机分配模拟

game2_result = fortune.T                    # 转置后得到结果数据 → 列为每一个人的id，行为每一轮的财富分配结果   

endtime = time.time()                       # 模型结束时间

print('模型总共用时%i秒' % (endtime - starttime))
#some analysis

lastround=pd.DataFrame({'money':game2_result.iloc[-1]}).sort_values('money',ascending = False).reset_index()

lastround['money_percent']=lastround['money']/lastround['money'].sum()

lastround['money_cum']=lastround['money_percent'].cumsum()

df=lastround

#

a=df.iloc[20]["money_cum"]

print('最后一轮中，20%%的人掌握了%.1f%%的财富'%(a*100))



#

stats=game2_result.iloc[-1].describe()

print('最后一轮中，最富的人有%i元,'%stats['max'],"相比于初始财富，翻了%.1f倍"%(stats['max']/100))



#

a=len(df[df['money']<100])/100 #财富少于100的比例

print('%.1f%%的人财富缩水'%(a*100))
#查看逆袭情况，第7k次负值玩家，在经过另一个7k次后，则此刻查看财富情况，将财富值为负的标记成“破产

# （财富值从负到正为逆袭）

r7000=pd.DataFrame({'money':game2_result.iloc[7000].sort_values(ascending = False).reset_index()[7000],\

                    'id': game2_result.iloc[7000].sort_values(ascending = False).reset_index()['id'],\

                    'color':'steelblue'})

r7000['color'][r7000['money']<0]='red'



negative= r7000[r7000['money']<0].id.tolist() #第7k轮负资产的人



#bar chart



for i in list(range(7000,10001,500)):

    datai = pd.DataFrame({'money':game2_result.iloc[i],'color':'steelblue'})

    

    datai['color'].loc[negative] = 'red'

    datai=datai.sort_values('money',ascending=False).reset_index()

    plt.figure(figsize=(10,6))

    plt.bar(datai.index, datai["money"], color=datai['color'],alpha = 0.7,width = 0.7)

    plt.title('round %i' %i)

r7000neg=datai[datai['id'].isin(negative)] #r7000曾负债的人

r7kneg1wpos=r7000neg[r7000neg['money']>0]  #r7000曾负债的人

r7kneg1wpos

print(int(3000/365),'年时间内','%i 个破产的人中，只有%i的人逆袭了，但都没有暴富'%(len(r7000neg),len(r7kneg1wpos),))
# 设置概率

person_p = [1/100.2 for i in range(0,100)]

for i in range(20):

    person_p[i] = 1.01/100.2 #可以方程求解20个人和另外800个人的概率



def game3(data, roundi):

    round_i = pd.DataFrame({'pre_round':data[roundi-1],'lost':1}) # 设定每轮分配财富之前的情况

    choice_i = pd.Series(np.random.choice(person_n,100,p = person_p))

    gain_i = pd.DataFrame({'gain':choice_i.value_counts()})       # 这一轮中每个人随机指定给“谁”1元钱，并汇总这一轮每个人的盈利情况

    round_i = round_i.join(gain_i)

    round_i.fillna(0,inplace = True)

    return round_i['pre_round'] -  round_i['lost'] + round_i['gain'] # 合并数据，得到这一轮财富分配的结果



# 设定初始参数：游戏玩家100人，起始资金100元

person_n = [x for x in range(1,101)]

fortune = pd.DataFrame([100 for i in range(100)], index = person_n)

fortune.index.name = 'id'



#迭代15330次，模拟财富分配 

starttime = time.time()                     # 模型开始时间

n=15330

for round in range(1,15331):

    fortune[round] = game3(fortune,round)   # 进行10000轮随机分配模拟

game3_result = fortune.T                    # 转置后得到结果数据 → 列为每一个人的id，行为每一轮的财富分配结果   

endtime = time.time()                       # 模型结束时间

print('模型总共用时%i秒' % (endtime - starttime))    
n=15330

df=pd.DataFrame({'money':game3_result.iloc[n].sort_values(ascending=False).reset_index()[n],\

                'id':game3_result.iloc[9].sort_values(ascending=False).reset_index()['id'],

               'color':'steelblue'})

df['color'][df['id'].isin(list(range(1,21)))]='orange'

#data0['color'].loc[[1,11,21,31,41,51,61,71,81,91]] = 'red'

#绘图

plt.figure(figsize=(10,6))

plt.bar(df.index,df['money'],color=df['color'],alpha=0.7)

plt.title('model3: work 1% harder than others')
#conclusion



#50岁财富为正的比例

a=len(df[df['money']>0])/100

print('社会平均水平的正资产率为%.0f%%'%(a*100))

df20=df[df['id'].isin(list(range(1,21)))]

a1=len(df20[df20['money']>0])/20

print('努力的人的正资产率为%.0f%%\n'%(a1*100))



#赚了的人

b=len(df[df['money']>100])/100

print('社会平均水平的资产>第一桶金100的概率%.2f'%b)

b1=len(df20[df20['money']>100])/20

print('努力的人最终资产>第一桶金100的概率%.2f'%b1)



print('\n社会财富的总体分布形态没有什么变化')
