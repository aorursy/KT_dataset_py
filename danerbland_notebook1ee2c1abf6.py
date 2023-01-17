# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.




df = pd.read_csv('../input/CandidateSummaryAction1.csv')

df['tot_dis'] = df['net_ope_exp'].str.replace("$","").str.replace(",","").str.replace("(","-").str.replace(")","")

df['net_ope_exp'] = df['net_ope_exp'].str.replace("$","").str.replace(",","").str.replace("(","-").str.replace(")","")

df['net_con'] = df['net_con'].str.replace("$","").str.replace(",","").str.replace("(","-").str.replace(")","")

df['cas_on_han_beg_of_per']= df['cas_on_han_beg_of_per'].str.replace("$","").str.replace(",","").str.replace("(","-").str.replace(")","")

df['cas_on_han_clo_of_per']= df['cas_on_han_clo_of_per'].str.replace("$","").str.replace(",","").str.replace("(","-").str.replace(")","")

df['tot_loa']= df['tot_loa'].str.replace("$","").str.replace(",","").str.replace("(","-").str.replace(")","")

df['can_loa'] = df['can_loa'].str.replace("$","").str.replace(",","").str.replace("(","-").str.replace(")","")

df['ind_ite_con']=df['ind_ite_con'].str.replace("$","").str.replace(",","").str.replace("(","-").str.replace(")","")

df['ind_uni_con']=df['ind_uni_con'].str.replace("$","").str.replace(",","").str.replace("(","-").str.replace(")","")

df['ind_con']=df['ind_con'].str.replace("$","").str.replace(",","").str.replace("(","-").str.replace(")","")





df.head(3)
winners = df[df.winner == 'Y']

swinners = winners[winners.can_off == "S"]

hwinners = winners[winners.can_off == "H"]
swexpenses = swinners['net_ope_exp']

swe = []

for row in swexpenses:

    swe.append(float(row))

plt.hist(swe, bins = 12, rwidth = 1, label = "Plot1")

plt.title('Senate Winning Campaigns (2016)')

plt.xlabel('Net Operating Expenses\n$(1,000,000)')

plt.ylabel('Winning Campaigns')

plt.show()

print("Minimum Winning Expenditures: " + str(min(swe)))

print("Maximum Winning Expenditures: " + str(max(swe)))

print("Mean Winning Expenditures: "+ str(np.mean(swe)))

print("Median Winning Expenditures: "+ str(np.median(swe)))

hwexpenses = hwinners['net_ope_exp']

hwe = []

for row in hwexpenses:

    row = row.replace("$", "")

    row = row.replace(" ", "")

    row = row.replace(",", "")

    hwe.append(float(row))

plt.hist(hwe, bins = 50, rwidth = 1, label = "Plot2")

plt.title('House Winning Campaigns (2016)')

plt.xlabel('Net Operating Expenses')

plt.ylabel('Winning Campaigns')

plt.show()

print("Minimum Winning Expenditures: " + str(min(hwe)))

print("Maximum Winning Expenditures: " + str(max(hwe)))

print("Mean Winning Expenditures: "+ str(np.mean(hwe)))

print("Median Winning Expenditures: "+ str(np.median(hwe)))
#look at the house districts.  first item is state name, second item is number of districts

us_states = (['AK', 1], ['AL',7],  ['AR',4], ['AZ', 9], ['CA', 53], ['CO', 7], ['CT', 5],[ 'DC', 0], ['DE', 1],  ['FL', 27], ['GA',14], \

                     ['HI', 2],  ['IA', 4], ['ID', 2], ['IL', 18], ['IN', 9], ['KS', 4], ['KY', 6], ['LA', 6], ['MA', 2], ['MD', 8], ['ME', 2], ['MI', 14],\

                     ['MN', 8], ['MO', 1], ['MS', 4], ['MT', 1], ['NC', 13], ['ND', 1], ['NE', 3], ['NH', 2], ['NJ', 12], ['NM', 3], ['NV', 4], ['NY', 27],\

                     ['OH', 16], ['OK', 5], ['OR', 5], ['PA', 18], ['RI', 2], ['SC', 7], ['SD', 1], ['TN', 9], ['TX', 36], ['UT', 4], ['VA', 11], ['VT', 1],\

                     ['WA', 10], ['WI', 8], ['WV', 3], ['WY', 1])



#For each state

for state in us_states:

    #isolate the house races in each state.

    statei = df[df.can_off_sta == state[0]]

    statei = statei[df.can_off == "H"]

    

    #isolate each district:

    for district in range(0, state[1]+1):

        districti = statei[statei.can_off_dis == district]

        

        #make sure the race was not empty (for at large positions, etc.)

        if districti.empty == False:



            #Isolate all of the losers in the district:        

            exp = np.array(districti.net_ope_exp[districti.winner !="Y"], dtype = 'f')

            con = np.array(districti.net_con[districti.winner !="Y"], dtype = 'f')

            

            #Isolate the winners separately so that we can distinguish them on the plot:

            wexp = np.array(districti.net_ope_exp[districti.winner =="Y"], dtype = 'f')

            wcon = np.array(districti.net_con[districti.winner =="Y"], dtype = 'f')

                        

            #build the plot:

            #plt.scatter(x = con, y = exp)

            #plt.scatter(x = wcon, y = wexp, c = "red")

            #plt.xlabel("Net Campaign Contributions")

            #plt.ylabel("Net Campaign Expenditures")

            #plt.title("House Campaigns in District: " + str(district) + "\n" + state[0])

            ##plt.show()

#INVESTIGATES Contributions, cash on hand beginning of period, difference between contributions and expenditures, and cash+contributions

us_states = (['AK', 1], ['AL',7],  ['AR',4], ['AZ', 9], ['CA', 53], ['CO', 7], ['CT', 5],[ 'DC', 0], ['DE', 1],  ['FL', 27], ['GA',14], \

                     ['HI', 2],  ['IA', 4], ['ID', 2], ['IL', 18], ['IN', 9], ['KS', 4], ['KY', 6], ['LA', 6], ['MA', 2], ['MD', 8], ['ME', 2], ['MI', 14],\

                     ['MN', 8], ['MO', 1], ['MS', 4], ['MT', 1], ['NC', 13], ['ND', 1], ['NE', 3], ['NH', 2], ['NJ', 12], ['NM', 3], ['NV', 4], ['NY', 27],\

                     ['OH', 16], ['OK', 5], ['OR', 5], ['PA', 18], ['RI', 2], ['SC', 7], ['SD', 1], ['TN', 9], ['TX', 36], ['UT', 4], ['VA', 11], ['VT', 1],\

                     ['WA', 10], ['WI', 8], ['WV', 3], ['WY', 1])



#Tracks the winning campaigns that also had the highest contributions, cash, contributions - expenditures, loans, and cash+con:

cmaxwinners = 0

cashwinners = 0

diffwinners = 0

loanwinners = 0

ccwinners = 0

#counts the races

races = 0



#For each state

for state in us_states:

    

    #isolate the house races in each state.

    statei = df[df.can_off_sta == state[0]]

    statei = statei[df.can_off == "H"]

    

    #isolate each district:

    for district in range(0, state[1]+1):

        districti = statei[statei.can_off_dis == district]

        

        #make sure the race was not empty (District 0 has both real and empty races, depending on the state)

        if districti.empty == False:

            

            #increment the race counter

            races += 1

        

            #Isolate expenses and contributions of all of the losers in the district:        

            exp = np.array(districti.net_ope_exp[districti.winner !="Y"], dtype = 'f')

            con = np.array(districti.net_con[districti.winner !="Y"], dtype = 'f')

            cash = np.array(districti.cas_on_han_beg_of_per[districti.winner !="Y"], dtype = 'f')

            diff = con - exp

            loan = np.array(districti.tot_loa[districti.winner !="Y"], dtype = 'f')

            cc = con + cash

            

            #Isolate the winners separately so that we can distinguish them on the plot:

            wexp = np.array(districti.net_ope_exp[districti.winner =="Y"], dtype = 'f')

            wcon = np.array(districti.net_con[districti.winner =="Y"], dtype = 'f')

            wcash = np.array(districti.cas_on_han_beg_of_per[districti.winner =="Y"], dtype = 'f')

            wdiff = wexp - wcon

            wloan = np.array(districti.tot_loa[districti.winner =="Y"], dtype = 'f')

            wcc = wcon + wcash

            

            #if there were other candidates:

            if con.size != 0:

                #if the winner has the highest contributions, increment counter. Also deals with NaN occurances

                if wcon >= max(exp) or np.isnan(max(con)) == True:

                    cmaxwinners += 1

            #if there was only one candidate, they must have the highest contributions as well:

            elif con.size == 0:

                cmaxwinners += 1

                



            if cash.size != 0:

                if wcash >= max(cash) or np.isnan(max(cash)) == True:

                    cashwinners += 1



            elif cash.size == 0:

                cashwinners += 1

                

                

            if diff.size != 0:

                if wdiff >= max(diff) or np.isnan(max(diff)) == True:

                    diffwinners += 1



            elif diff.size == 0:

                diffwinners += 1      

                

                

            if cc.size != 0:

                if wcc >= max(cc) or np.isnan(max(cc)) == True:

                    ccwinners += 1



            elif cc.size == 0:

                ccwinners += 1      

                



          



print("Percentage of winners with the highest contributions: " + str(cmaxwinners * 100/races))



print("Percentage of winners with the highest initial cash on hand: " + str(cashwinners * 100/races))



print("Percentage of winners with the highest net income (contributions - expenses): " + str(diffwinners * 100/races))



print("Percentage of winners with the highest cash on hand + contributions: " + str(ccwinners * 100/races))