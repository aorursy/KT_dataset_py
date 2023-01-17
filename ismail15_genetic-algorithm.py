import random
import numpy as np
import pandas as pd
no_of_gene=20
no_of_population=30

def generate_chromosome(ch_length,no_of_ch):
    population=[]
    for i in range(no_of_ch):
        ch=""
        for j in range(ch_length):
            gene=str(random.randint(0,1))
            ch+=gene
#         print("chromoseome no "+ str(i+1) +": "+ ch)  
        population.append(ch)
    return population
population=[]
population=generate_chromosome(no_of_gene,no_of_population)

def selection(population):
    x_value=[]
    f_x_value=[]
    pselect=[]
    expected_count=[]
    actual_count=[]
    
    for i in range(len(population)):
        decimal_value=int(population[i],2)
        fun_value=1/decimal_value
    
        x_value.append(decimal_value)
        f_x_value.append(1/decimal_value)
    
    sum_f_x=sum(f_x_value)
    avg_f_x=sum_f_x/len(population)
    
    for j in range(len(population)):
        p_select=f_x_value[j]/sum_f_x
        pselect.append(p_select)
        
    sum_p_select=sum(pselect)
    avg_p_select=sum_p_select/len(population)
    
    for k in range(len(population)):
        e_count=pselect[k]/avg_p_select
        expected_count.append(e_count)
        a_count=round(e_count)
        actual_count.append(a_count)
        
        
    sum_a_count=sum(actual_count)
    avg_a_count=sum_a_count/len(population)
  
    return x_value,f_x_value,pselect,expected_count,actual_count
    

def selection_output(x_value,f_x_value,pselect,expected_count,actual_count):
    selection_1 = pd.DataFrame()
    selection_1['initial population']=population
    selection_1['x value']=x_value
    selection_1['f(x)=1/x']=f_x_value
    selection_1['p select']=pselect
    selection_1['expected count']=expected_count
    selection_1['actual count']=actual_count
    print(selection_1)

def met_pool_gen(population,actual_count):
    selected_ch=[]
    for i in range(len(population)):
        for j in range(actual_count[i]):
            selected_ch.append(population[i])
      
    return selected_ch
def crossover(population,actual_count):
    mating_pool=met_pool_gen(population,actual_count)
    ch_index=[]
    new_population={}
    mate_with={}
    c_site={}
    new_x={}
    new_f_x={}
    
    for i in range(len(mating_pool)):
        ch_index.append(i)
    no_of_pair=int(len(ch_index)/2) 
    
    for j in range(no_of_pair):
        #crossover pair selection
        index_1=random.choice(ch_index)
        ch_index.remove(index_1)
        index_2=random.choice(ch_index)
        ch_index.remove(index_2)
        
        
        
        #crossover pair
        ch_1=mating_pool[index_1]
        ch_2=mating_pool[index_2]

        
        
        #crossover site selection
        crossover_site=random.randint(0,no_of_gene-2)
        
        #offspring generation
        offspring_1=""
        offspring_2=""
        
        offspring_1=ch_2[:crossover_site]
        offspring_1+=ch_2[crossover_site:]
        
        offspring_2=ch_1[:crossover_site]
        offspring_2+=ch_1[crossover_site:]
        
        
        new_population[index_1]=offspring_1
        new_population[index_2]=offspring_2
        
        new_x[index_1]=int(offspring_1,2)
        new_x[index_2]=int(offspring_2,2)
        
        new_f_x[index_1]=1/new_x[index_1]
        new_f_x[index_2]=1/new_x[index_2]
        
        c_site[index_1]=crossover_site
        c_site[index_2]=crossover_site
        
        mate_with[index_1]=index_2
        mate_with[index_2]=index_1
    
    
    temp=sorted(c_site.items())
    ch_no=[lis[0] for lis in temp]
    max_v=max(ch_no)
    miss_index=0
    for i in range(len(ch_no)):
        if i!=ch_no[i]:
            miss_index=i
            break
    print(type(c_site))        
    mate_with[miss_index]='-'
    c_site[miss_index]='-'
    new_population[miss_index]=mating_pool[miss_index]
    new_x[miss_index]='-'
    new_f_x[miss_index]='-'
    
    new_population=sorted(new_population.items())
    mate_with=sorted(mate_with.items())
    c_site=sorted(c_site.items())
    new_x=sorted(new_x.items())
    new_f_x=sorted(new_f_x.items())
    
    mate_with = [lis[1] for lis in mate_with] 
    c_site=[lis[1] for lis in c_site]
    new_population=[lis[1] for lis in new_population]
    new_x=[lis[1] for lis in new_x]
    new_f_x=[lis[1] for lis in new_f_x]
    
    
    
    
    return mating_pool,mate_with,c_site,new_population,new_x,new_f_x


def crossover_output(mating_pool,mate_with,c_site,new_population,new_x,new_f_x,gen_no):
    generation= pd.DataFrame()
    generation['Mating pool']=mating_pool
    generation['mate_with']=mate_with
    generation['c_site']=c_site
    generation['new_population']=new_population
    generation['new_x']=new_x
    generation['new_f_x']=new_f_x
    print("generation :"+str(gen_no))
    print(generation)
    





number_of_gen=4

for i in range(number_of_gen):
    x_value=[]
    f_x_value=[]
    pselect=[]
    expected_count=[]
    actual_count=[]
    x_value,f_x_value,pselect,expected_count,actual_count=selection(population)
    selection_output( x_value,f_x_value,pselect,expected_count,actual_count)
    
    mating_pool=[]
    mate_with=[]
    c_site=[]
    new_population=[]
    new_x=[]
    new_f_x=[]
    mating_pool,mate_with,c_site,new_population,new_x,new_f_x=crossover(population,actual_count)
    crossover_output(mating_pool,mate_with,c_site,new_population,new_x,new_f_x,i+1)
    population=new_population
    
    
    
    
    