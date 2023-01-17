import random



random.seed(3228)

nfold=2



#read orders

fr = open("../input/orders.csv", 'r')

fr.readline()# skip header

lines=fr.readlines()

valid_order_ids=[[] for i in range(nfold)]

print("Total {} lines in orders.csv".format(len(lines)))



fold_csvs=[]

for fold in range(nfold):

    outcsv = open("orders_valid_{}.csv".format(fold), 'w')

    outcsv.writelines("order_id,user_id,eval_set,order_number,order_dow,order_hour_of_day,days_since_prior_order"+"\n")

    fold_csvs.append(outcsv)



for i,line in enumerate(lines):

    datas=line.replace("\n","").split(",")

    order_id=datas[0]

    eval_set=datas[2]

    if(eval_set=='train'):

        randNum=random.randint(1,1000)

        for j in range(nfold):

            if(randNum<=(j+1)*1000/(nfold) and randNum>(j)*1000/(nfold)):

                valid_order_ids[j].append(order_id)

                datas[2]='valid'

                outline=','.join(datas)

                fold_csvs[j].writelines(outline+"\n")

            else:

                outline=','.join(datas)

                fold_csvs[j].writelines(outline+"\n")

        

    else:

        outline=','.join(datas)

        for j in range(nfold):

            fold_csvs[j].writelines(outline+"\n")
orders={}         

#read train orders

fr = open("../input/order_products__train.csv", 'r')

fr.readline()# skip header

lines=fr.readlines()

for i,line in enumerate(lines):

    datas=line.replace("\n","").split(",")

    order_id=datas[0]

    product_id=datas[1]

    reorderer=int(datas[3])

    if(order_id not in orders):

        orders[order_id]=[]

    if(reorderer==1):

        orders[order_id].append(product_id)



for fold in range(nfold):

    outcsv = open("orders_valid_label_{}.csv".format(fold), 'w')

    outcsv.writelines("order_id,label"+"\n")

    for order_id in valid_order_ids[fold]:

        if(len(orders[order_id])==0):

            orders[order_id].append('None')

        datas=[order_id,' '.join(orders[order_id])]

        outcsv.writelines(','.join(datas)+"\n")
from sklearn.metrics import f1_score        

for fold in range(nfold):

    f1_scores=[]

    for order_id in valid_order_ids[fold]:

        y_pred_labels=['24852'] # Use Banana as predict, you must replace it with your own prediction

        y_true_labels=orders[order_id]

        labels=list(set(y_pred_labels)|set(y_true_labels))

        

        y_pred=[]

        y_true=[]

        for label in labels:

            if(label in y_true_labels):

                y_true.append(1)

            else:

                y_true.append(0)

            if(label in y_pred_labels):

                y_pred.append(1)

            else:

                y_pred.append(0)

        score=f1_score(y_true, y_pred)

        f1_scores.append(score)

    print("F1 Score = {} for all Banana's prediction".format(sum(f1_scores)/len(f1_scores)))