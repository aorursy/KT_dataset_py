import zipfile
%matplotlib inline 
import numpy as np 
import scipy as sp 
import matplotlib as mpl
import matplotlib.cm as cm 
import matplotlib.pyplot as plt
import pandas as pd 
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns
sns.set(style="whitegrid")
import warnings
warnings.filterwarnings('ignore')
events = pd.read_csv("../input/events.csv")
game_info=pd.read_csv("../input/ginf.csv")
events.head()
events.info()
encoding = pd.read_table('../input/dictionary.txt', delim_whitespace=False, names=('num','events'))
event_type=encoding[1:13]
event_type_2=encoding[14:18]
side=encoding[19:21]
shot_place=encoding[22:35]
shot_outcome=encoding[36:40]
location=encoding[41:60]
bodypart=encoding[61:64]
assist_method=encoding[65:70]
situition=encoding[71:75]
event_type
event_type_2
side
shot_place
shot_outcome
location
bodypart
assist_method
situition
#Note that 2 refers to set-piece excluding direct free kicks
goals=events[events["is_goal"]==1]
fig=plt.figure(figsize=(8,6))
plt.hist(goals.time,width=1,bins=100,color="green")   #100 so 1 bar per minute
plt.xlabel("Minutes")
plt.ylabel("Number of goals")
plt.title("Number of goals against Time during match")
fig=plt.figure(figsize=(8,6))
plt.hist(goals[goals["side"]==1]["time"],width=1,bins=100,color="cyan",label="home goals")   
plt.hist(goals[goals["side"]==2]["time"],width=1,bins=100,color="grey",label="away goals") 
plt.xlabel("Minutes")
plt.ylabel("Number of goals")
plt.legend()
plt.title("Number of goals (by home/away side) against Time during match")
plt.subplot(2,1,1)
labels=["Right Foot","Left Foot","Headers"]
sizes=[goals[goals["bodypart"]==1].shape[0],goals[goals["bodypart"]==2].shape[0],goals[goals["bodypart"]==3].shape[0]]
colors=["cyan","grey","pink"]
plt.pie(sizes,labels=labels,colors=colors,autopct='%1.1f%%',startangle=60)
plt.axis('equal')
plt.title("Percentage of bodyparts for goals",fontname="Times New Roman Bold",fontsize=14,fontweight="bold")
fig=plt.gcf() 
fig.set_size_inches(10,10)


plt.subplot(2,1,2)
plt.hist(goals[goals["bodypart"]==1]["time"],width=1,bins=100,color="cyan",label="Right foot")   
plt.hist(goals[goals["bodypart"]==2]["time"],width=1,bins=100,color="grey",label="Left foot") 
plt.hist(goals[goals["bodypart"]==3]["time"],width=1,bins=100,color="pink",label="Headers") 
plt.xlabel("Minutes")
plt.ylabel("Number of goals")
plt.legend()
plt.title("Number of goals (by body parts) against Time during match",fontname="Times New Roman Bold",fontsize=14,fontweight="bold")
plt.tight_layout()

plt.subplot(2,1,1)
labels=["Open Play", "Set Piece(Excluding direct Free kick)","Corners","Direct Free Kick"]
sizes=[goals[goals["situation"]==1].shape[0],goals[goals["situation"]==2].shape[0],goals[goals["situation"]==3].shape[0],goals[goals["situation"]==4].shape[0]]
colors=["cyan","grey","blue","yellow"]
plt.pie(sizes,labels=labels,colors=colors,autopct='%1.1f%%',startangle=60)
plt.axis('equal')
plt.title("Percentage of each situation for goals",fontname="Times New Roman Bold",fontsize=14,fontweight="bold")
fig=plt.gcf() 
fig.set_size_inches(10,10)

plt.subplot(2,1,2)
plt.hist(goals[goals["situation"]==1]["time"],width=1,bins=100,color="cyan",label="Open play")   
plt.hist(goals[goals["situation"]==2]["time"],width=1,bins=100,color="grey",label="Set Piece (excluding direct FreeKick)") 
plt.hist(goals[goals["situation"]==3]["time"],width=1,bins=100,color="blue",label="Corners") 
plt.hist(goals[goals["situation"]==4]["time"],width=1,bins=100,color="yellow",label="Direct Free Kick") 
plt.xlabel("Minutes")
plt.ylabel("Number of goals")
plt.legend()
plt.title("Number of goals (by situations) against Time during match",fontname="Times New Roman Bold",fontsize=14,fontweight="bold")
plt.tight_layout()
for i in range(20):
    if sum(goals["location"]==i)==0:
        print(i)
diff_angle_goals=sum(goals["location"]==6)+sum(goals["location"]==7)+sum(goals["location"]==8)
long_range_goals=sum(goals["location"]==16)+sum(goals["location"]==17)+sum(goals["location"]==18)
box_goals=sum(goals["location"]==3)+sum(goals["location"]==9)+sum(goals["location"]==11)+sum(goals["location"]==15)
close_range_goals=sum(goals["location"]==10)+sum(goals["location"]==12)+sum(goals["location"]==13)
penalties=sum(goals["location"]==14)
not_recorded=sum(goals["location"]==19)

labels=["Long Range goals","Difficult angle goals","Goals from around the box","Close range goals","Penalties","Not recorded"]
sizes=[long_range_goals,diff_angle_goals,box_goals,close_range_goals,penalties,not_recorded]
colors=["gray","yellow","aqua","coral","red","violet"]
plt.pie(sizes,colors=colors,autopct='%1.1f%%',startangle=60,pctdistance=0.8,radius=3)
plt.axis('equal')
plt.title("Percentage of each location for goals",fontname="Times New Roman Bold",fontsize=18,fontweight="bold")
plt.legend(labels)
fig=plt.gcf() 
fig.set_size_inches(12,8)
plt.show()
labels=["None","Pass","Cross","Headed Pass","Through Ball"]
sizes=[sum(goals["assist_method"]==0),sum(goals["assist_method"]==1),sum(goals["assist_method"]==2),sum(goals["assist_method"]==3),sum(goals["assist_method"]==4)]
colors=["palegreen","yellow","aqua","coral","plum"]
plt.pie(sizes,labels=labels,colors=colors,autopct='%1.1f%%',startangle=60)
plt.axis('equal')
plt.title("Percentage of each assist type for goals",fontname="Times New Roman Bold",fontsize=18,fontweight="bold")
fig=plt.gcf()  #gcf --> current figure
fig.set_size_inches(8,6)
plt.show()
substitution=events[events["event_type"]==7]
fig=plt.figure(figsize=(8,6))
plt.hist(substitution.time,width=1,bins=100,color="green")   
plt.xlabel("Minutes")
plt.ylabel("Number of substitutions")
plt.title("Number of substitutions against Time during match")
plt.subplot(2,1,1)
labels=["Home","Away"]
sizes=[substitution[substitution["side"]==1].shape[0],substitution[substitution["side"]==2].shape[0]]
colors=["green","aqua"]
plt.pie(sizes,labels=labels,colors=colors,autopct='%1.1f%%',startangle=90)
plt.axis('equal')
plt.title("Percentage of substitutions by Home/Away",fontname="Times New Roman Bold",fontsize=14,fontweight="bold")
fig=plt.gcf() 
fig.set_size_inches(10,10)

plt.subplot(2,1,2)
plt.hist(substitution[substitution["side"]==1].time,width=1,bins=100,color="green",label="home")
plt.hist(substitution[substitution["side"]==2].time,width=1,bins=100,color="aqua",label="away")
plt.xlabel("Minutes")
plt.ylabel("Number of substitutions")
plt.legend()
plt.title("Number of substitutions (home/away) against Time during match",fontname="Times New Roman Bold",fontsize=18,fontweight="bold")
yel_card=events[events["event_type"]==4]
yel_card.shape
plt.figure(figsize=(8,6))
plt.hist(yel_card.time,width=8,bins=9,color="yellow")   
plt.xlabel("Minutes")
plt.ylabel("Number of yellow cards")
plt.title("Number of yellow cards against Time during match",fontname="Times New Roman Bold",fontsize=14,fontweight="bold")

yel_home=yel_card[yel_card["side"]==1].shape[0]
yel_away=yel_card[yel_card["side"]==2].shape[0]

yel_combi=pd.DataFrame({"home":yel_home,"away":yel_away},index=["Yellow cards"])
yel_combi.plot(kind="bar")
plt.title("Number of yellow cards (Home/Away)",fontname="Times New Roman Bold",fontsize=14,fontweight="bold")
yel_combi
sec_yellow=events[events["event_type"]==5]
red=events[events["event_type"]==6]
reds=[sec_yellow,red]
red_cards=pd.concat(reds)
red_cards.event_type.unique()
fig=plt.figure(figsize=(8,6))
plt.hist(red_cards.time,width=8,bins=9,color="red")   
plt.xlabel("Minutes")
plt.ylabel("Number of red cards")
plt.title("Number of red cards against Time during match",fontname="Times New Roman Bold",fontsize=14,fontweight="bold")
red_home=red_cards[red_cards["side"]==1].shape[0]
red_away=red_cards[red_cards["side"]==2].shape[0]

red_combi=pd.DataFrame({"home":red_home,"away":red_away},index=["Red cards"])
red_combi.plot(kind="bar")
plt.title("Number of Red cards (Home/Away)",fontname="Times New Roman Bold",fontsize=14,fontweight="bold")
red_combi
penalties=events[events["location"]==14]
penalties.head()
for i in range(14):
    if sum(penalties["shot_place"]==i)==0:
        print(i)
top_left=sum(penalties["shot_place"]==12)
bot_left=sum(penalties["shot_place"]==3)
top_right=sum(penalties["shot_place"]==13)
bot_right=sum(penalties["shot_place"]==4)
centre=sum(penalties["shot_place"]==5)+sum(penalties["shot_place"]==11)
missed=sum(penalties["shot_place"]==1)+sum(penalties["shot_place"]==6)+sum(penalties["shot_place"]==7)+sum(penalties["shot_place"]==8)+sum(penalties["shot_place"]==9)+sum(penalties["shot_place"]==10)

labels_pen=["top left","bottom left","centre","top right","bottom right","missed"]
num_pen=[top_left,bot_left,centre,top_right,bot_right,missed]
colors_pen=["aqua","royalblue","yellow","violet","m","red"]
plt.pie(num_pen,labels=labels_pen,colors=colors_pen,autopct='%1.1f%%',startangle=60,explode=(0,0,0,0,0,0.2))
plt.axis('equal')
plt.title("Percentage of each placement of penalties",fontname="Times New Roman Bold",fontsize=18,fontweight="bold")
fig=plt.gcf()  
fig.set_size_inches(8,6)
plt.show()
scored_pen=penalties[penalties["is_goal"]==1]
pen_rightfoot=scored_pen[scored_pen["bodypart"]==1].shape[0]
pen_leftfoot=scored_pen[scored_pen["bodypart"]==2].shape[0]
penalty_combi=pd.DataFrame({"right foot":pen_rightfoot,"left foot":pen_leftfoot},index=["Scored"])
penalty_combi.plot(kind="bar")
plt.title("Penalties scored (Right/Left foot)",fontname="Times New Roman Bold",fontsize=14,fontweight="bold")
penalty_combi
missed_pen=penalties[penalties["is_goal"]==0]
missed_right=missed_pen[missed_pen["bodypart"]==1].shape[0]
missed_left=missed_pen[missed_pen["bodypart"]==2].shape[0]
missed_combi=pd.DataFrame({"right foot":missed_right,"left foot":missed_left},index=["Missed"])
missed_combi.plot(kind="bar")
plt.title("Penalties Missed (Right/Left foot)",fontname="Times New Roman Bold",fontsize=14,fontweight="bold")
missed_combi
total_missed=[missed_right, missed_left]
total_scored=[pen_rightfoot, pen_leftfoot]
plt.bar(penalties["bodypart"].unique(),total_missed,color="black",label="Missed")
plt.bar(penalties["bodypart"].unique(),total_scored,color="red",bottom=total_missed,label="Scored")
plt.xticks(penalties["bodypart"].unique(),["Right foot","Left foot"])
plt.ylabel("Frequency")
plt.legend()
plt.title("Penalties scored/missed among right/left foot",fontname="Times New Roman Bold",fontsize=14,fontweight="bold")
plt.show()
total_pen=penalty_combi.append(missed_combi)
total_pen
conversion_left_foot=total_pen["left foot"]["Scored"]/(total_pen["left foot"]["Scored"]+total_pen["left foot"]["Missed"])
conversion_right_foot=total_pen["right foot"]["Scored"]/(total_pen["right foot"]["Scored"]+total_pen["right foot"]["Missed"])
print("Left foot conversion rate is: " + str("%.2f" % conversion_left_foot))
print("Right foot conversion rate is: " + str("%.2f" % conversion_right_foot))
def pen_stats(player):
    player_pen=penalties[penalties["player"]==player]
    right_attempt=player_pen[player_pen["bodypart"]==1]
    right_attempt_scored=right_attempt[right_attempt["is_goal"]==1].shape[0]
    right_attempt_missed=right_attempt[right_attempt["is_goal"]==0].shape[0]
    left_attempt=player_pen[player_pen["bodypart"]==2]
    left_attempt_scored=left_attempt[left_attempt["is_goal"]==1].shape[0]
    left_attempt_missed=left_attempt[left_attempt["is_goal"]==0].shape[0]
    scored=pd.DataFrame({"right foot":right_attempt_scored,"left foot":left_attempt_scored},index=["Scored"])
    missed=pd.DataFrame({"right foot":right_attempt_missed,"left foot":left_attempt_missed},index=["Missed"])
    combi=scored.append(missed)
    return combi
    
penalties[penalties["shot_place"].isnull()].is_goal.unique()
pen_stats("eden hazard")
def pen_full_stats(player):
    player_pen=penalties[penalties["player"]==player]
    scored_pen=player_pen[player_pen["is_goal"]==1]
    missed_pen=player_pen[player_pen["is_goal"]==0]
    
    top_left_rightfoot=scored_pen[scored_pen["shot_place"]==12][scored_pen["bodypart"]==1].shape[0]
    top_left_leftfoot=scored_pen[scored_pen["shot_place"]==12][scored_pen["bodypart"]==2].shape[0]
    bot_left_rightfoot=scored_pen[scored_pen["shot_place"]==3][scored_pen["bodypart"]==1].shape[0]
    bot_left_leftfoot=scored_pen[scored_pen["shot_place"]==3][scored_pen["bodypart"]==2].shape[0]
    top_right_rightfoot=scored_pen[scored_pen["shot_place"]==13][scored_pen["bodypart"]==1].shape[0]
    top_right_leftfoot=scored_pen[scored_pen["shot_place"]==13][scored_pen["bodypart"]==2].shape[0]
    bot_right_rightfoot=scored_pen[scored_pen["shot_place"]==4][scored_pen["bodypart"]==1].shape[0]
    bot_right_leftfoot=scored_pen[scored_pen["shot_place"]==4][scored_pen["bodypart"]==2].shape[0]
    centre_rightfoot=scored_pen[scored_pen["shot_place"]==5][scored_pen["bodypart"]==1].shape[0]+scored_pen[scored_pen["shot_place"]==11][scored_pen["bodypart"]==1].shape[0]
    centre_leftfoot=scored_pen[scored_pen["shot_place"]==5][scored_pen["bodypart"]==2].shape[0]+scored_pen[scored_pen["shot_place"]==11][scored_pen["bodypart"]==2].shape[0]
    scored_without_recorded_loc_rightfoot=scored_pen[scored_pen["shot_place"].isnull()][scored_pen["bodypart"]==1].shape[0]
    scored_without_recorded_loc_leftfoot=scored_pen[scored_pen["shot_place"].isnull()][scored_pen["bodypart"]==2].shape[0]
    missed_rightfoot=missed_pen[missed_pen["bodypart"]==1].shape[0]
    missed_leftfoot=missed_pen[missed_pen["bodypart"]==2].shape[0]
    
    right_foot=pd.DataFrame({"Top Left Corner":top_left_rightfoot,"Bottom Left Corner":bot_left_rightfoot,"Top Right Corner":top_right_rightfoot,"Bottom Right Corner":bot_right_rightfoot,"Centre":centre_rightfoot,"Unrecorded placement":scored_without_recorded_loc_rightfoot,"Missed":missed_rightfoot},index=["Right Foot attempt"])
    left_foot=pd.DataFrame({"Top Left Corner":top_left_leftfoot,"Bottom Left Corner":bot_left_leftfoot,"Top Right Corner":top_right_leftfoot,"Bottom Right Corner":bot_right_leftfoot,"Centre":centre_leftfoot,"Unrecorded placement":scored_without_recorded_loc_leftfoot,"Missed":missed_leftfoot},index=["Left Foot attempt"])
    
    fullstats=right_foot.append(left_foot)
    fullstats=fullstats[["Top Right Corner","Bottom Right Corner","Top Left Corner","Bottom Left Corner","Centre","Unrecorded placement","Missed"]]
    return fullstats
pen_full_stats("eden hazard")
game_info.head()
game_info.info()
def odds_pred_result(odd_h,odd_d,odd_a):
    if odd_h<odd_d and odd_h<odd_a:
        return("Home Win")
    elif odd_d<odd_h and odd_d<odd_a:
        return("Draw")
    elif odd_a<odd_d and odd_a<odd_h:
        return("Away Win")
def actual_result(fthg,ftag):
    if fthg>ftag:
        return("Home Win")
    elif fthg==ftag:
        return("Draw")
    elif fthg<ftag:
        return("Away Win")
    
def actual_result_encode(fthg,ftag):
    if fthg>ftag:
        return (1)
    elif fthg==ftag:
        return (2)
    elif fthg<ftag:
        return (3)
def check_pred(data):
    correct=0
    wrong=0
    for i in range(1,data.shape[0]+1):
        odd_h=data[i-1:i]["odd_h"].item()
        odd_d=data[i-1:i]["odd_d"].item()
        odd_a=data[i-1:i]["odd_a"].item()
        fthg=data[i-1:i]["fthg"].item()
        ftag=data[i-1:i]["ftag"].item()
        oddsresult=odds_pred_result(odd_h,odd_d,odd_a)
        actresult=actual_result(fthg,ftag)
        if oddsresult==actresult:
            correct+=1
        else:
            wrong+=1
    return(str("%.2f"%(correct/(correct+wrong)))+str("% correct"))
check_pred(game_info)
x_var=game_info.iloc[:,9:14]
x_var.head()
x_var=game_info.iloc[:,9:14]
result=[]
for i in range(1,game_info.shape[0]+1):
    result.append(actual_result_encode(game_info[i-1:i]["fthg"].item(),game_info[i-1:i]["ftag"].item()))
y=pd.DataFrame(result)
x_var["diff_h_d"]=abs(x_var["odd_h"]-x_var["odd_d"])
x_var["diff_d_a"]=abs(x_var["odd_d"]-x_var["odd_a"])
x_var["diff_h_a"]=abs(x_var["odd_h"]-x_var["odd_a"])
x_var=x_var.drop(["fthg","ftag"],axis=1)
x_var.tail()
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_var,y,test_size=0.2,random_state=0)
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
def acc_score(model):
    return np.mean(cross_val_score(model,x_train,y_train,cv=k_fold,scoring="accuracy"))
from sklearn.metrics import accuracy_score
def norm_score(model):
    return (accuracy_score(y_train,model.predict(x_train)))
from sklearn.metrics import confusion_matrix
def confusion_matrix_model(model_used):
    cm=confusion_matrix(y_train,model_used.predict(x_train))
    cm=pd.DataFrame(cm)
    cm.columns=["Predicted Home Win","Predicted Draw","Predicted Away Win"]
    cm.index=["Actual Home Win","Actual Draw","Actual Away Win"]
    return cm
log_reg=LogisticRegression()
log_reg.fit(x_train,y_train)
print("Logistic Regression Accuracy (without cross validation): "+str(norm_score(log_reg)))
print("Logistic Regression Cross Validation Accuracy: "+str(acc_score(log_reg)))
confusion_matrix_model(log_reg)
Dec_tree=DecisionTreeClassifier()
Dec_tree.fit(x_train,y_train)
print("Decision Tree Regression Accuracy (without cross validation): "+str(norm_score(Dec_tree)))
print("Decision Tree Regression Cross Validation Accuracy: "+str(acc_score(Dec_tree)))
confusion_matrix_model(Dec_tree)

KNN=KNeighborsClassifier(n_neighbors=20)
KNN.fit(x_train,y_train)
print("KNN Regression Accuracy (without cross validation): "+str(norm_score(KNN)))
print("KNN Regression Cross Validation Accuracy: "+str(acc_score(KNN)))
confusion_matrix_model(KNN)
svm_reg=svm.SVC()
svm_reg.fit(x_train,y_train)
print("SVM Regression Accuracy (without cross validation): "+str(norm_score(svm_reg)))
print("SVM Regression Cross Validation Accuracy: "+str(acc_score(svm_reg)))
confusion_matrix_model(KNN)
naive_bayes=MultinomialNB()
naive_bayes.fit(x_train,y_train)
print("Multinomial Naive Bayes Regression Accuracy (without cross validation): "+str(norm_score(naive_bayes)))
print("Multinomial Naive Bayes Regression Cross Validation Accuracy: "+str(acc_score(naive_bayes)))
confusion_matrix_model(naive_bayes)