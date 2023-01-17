import pandas as pd
import numpy as np
np.random.seed(seed=1)
listItem = []
for a in range(50) :
    listItem.append([np.random.randint(1001,1011),np.random.randint(3001,3006)])
cart = pd.DataFrame(data=listItem,columns=["userId","ItemId"])
cart["ItemName"]=cart["ItemId"]
dicti={3001:"Pasta Gigi",
       3002:"Pembersih Mulut",
       3003:"Pembersih Wajah",
       3004:"Sabun Mandi",
       3005:"Handuk"}
cart["ItemName"] = cart["ItemName"].replace(dicti)
cart = cart.sort_values(by="userId")
cart.index = range(1,51)
cart = cart.drop_duplicates(subset=['userId', 'ItemId'], keep="first")
cart
cart.groupby(["userId"]).count()
#Get list of unique items
itemList=list(set(cart["ItemId"].tolist()))

#Get count of users
userCount=len(set(cart["userId"].tolist()))

#Create an empty data frame to store item affinity scores for items.
itemAffinity= pd.DataFrame(columns=('item1', 'item2', 'score'))
rowCount=0

#For each item in the list, compare with other items.
for ind1 in range(len(itemList)):
    
    #Get list of users who bought this item 1.
    item1Users = cart[cart.ItemId==itemList[ind1]]["userId"].tolist()
    #print("Item 1 ", item1Users)
    
    #Get item 2 - items that are not item 1 or those that are not analyzed already.
    for ind2 in range(ind1, len(itemList)):
        
        if ( ind1 == ind2):
            continue
       
        #Get list of users who bought item 2
        item2Users=cart[cart.ItemId==itemList[ind2]]["userId"].tolist()
        #print("Item 2",item2Users)
        
        #Find score. Find the common list of users and divide it by the total users.
        commonUsers= len(set(item1Users).intersection(set(item2Users)))
        score=commonUsers / userCount
        #print(commonUsers)

        #Add a score for item 1, item 2
        itemAffinity.loc[rowCount] = [itemList[ind1],itemList[ind2],score]
        rowCount +=1
        #Add a score for item2, item 1
        itemAffinity.loc[rowCount] = [itemList[ind2],itemList[ind1],score]
        rowCount +=1
        
#Check final result
itemAffinity.sort_values(by="score",ascending=False)
list(cart["ItemName"].unique())
# SearchName=input("Recommendations after the first item in cart:")

SearchName= 'Pembersih Mulut'

print("Recommendations after the first item in cart:",SearchName)
inv_map = {v: k for k, v in dicti.items()}


searchItem=inv_map[SearchName]
treshold=0


recoList=itemAffinity[itemAffinity["item1"]==searchItem]\
                                    [["item2","score"]]\
                                    .sort_values("score", ascending=False)
recoList.insert(1, "ProductName", recoList["item2"], allow_duplicates=False)
recoList["ProductName"] = recoList["ProductName"].replace(dicti)

print("\nWith"," score treshold     :",str(treshold),"\n\n", recoList[recoList["score"]>=treshold])


