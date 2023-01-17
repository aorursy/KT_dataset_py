# load the data
import pandas as pd
df = pd.read_csv('../input/train.csv',index_col='PassengerId')
# select females and masters (boys)
boy = (df.Name.str.contains('Master')) | ((df.Sex=='male') & (df.Age<13))
female = df.Sex=='female'
boy_or_female = boy | female

# no. females + boys on ticket
n_ticket = df[boy_or_female].groupby('Ticket').Survived.count()

# survival rate amongst females + boys on ticket
tick_surv = df[boy_or_female].groupby('Ticket').Survived.mean()
# function to create relevant features for test data
def create_features(frame):
    frame['Boy'] = (frame.Name.str.contains('Master')) | ((frame.Sex=='male') & (frame.Age<13))
    
    frame['Female'] = (frame.Sex=='female').astype(int)

    # if ticket exists in training data, fill NTicket with no. women+boys
    # on that ticket in the training data.
    frame['NTicket'] = frame.Ticket.replace(n_ticket)
    # otherwise NTicket=0
    frame.loc[~frame.Ticket.isin(n_ticket.index),'NTicket']=0

    # if ticket exists in training data, fill TicketSurv with
    # women+boys survival rate in training data  
    frame['TicketSurv'] = frame.Ticket.replace(tick_surv)
    # otherwise TicketSurv=0
    frame.loc[~frame.Ticket.isin(tick_surv.index),'TicketSurv']=0

    # return data frame only including features needed for prediction
    return frame[['Female','Boy','NTicket','TicketSurv']]

# predict survival for a passenger
def did_survive(row):
    if row.Female:
        # predict died if all women+boys on ticket died
        if (row.NTicket>0) and (row.TicketSurv==0):
            return 0
        # predict survived for all other women
        else:
            return 1
        
    elif row.Boy:
        # predict survived if all women+boys on ticket survived
        if (row.NTicket>0) and (row.TicketSurv==1):
            return 1
        # predict died for all other boys
        else:
            return 0
        
    else:
        # predict all men die
        return 0
# load test data
df_test = pd.read_csv('../input/test.csv',index_col='PassengerId')

# extract the features to use
X = create_features(df_test)

# predict test data
pred = X.apply(did_survive,axis=1)

# create submission file
pred = pd.DataFrame(pred) 
pred.rename(columns={0:'Survived'},inplace=True)
pred.to_csv('submission.csv')

print(pred.Survived.value_counts())
# passengers where I predict differently
# (based on comparison with Chris Deotte's output file)
id_nomatch = [910,929,1172]

# display these passengers
display(df_test.loc[id_nomatch])
# families in training data with member in test data that I predict differently
display(df.loc[df.Name.str.contains('Ilmakangas')])
display(df.loc[df.Name.str.contains('Cacic')])
display(df.loc[df.Name.str.contains('Oreskovic')])