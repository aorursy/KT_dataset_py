seed = 300
from math import *

import math

import numpy as np

import pandas as pd
Train = pd.read_csv("../input/hta-tagging/train.csv")

Train.head(5)
Test = pd.read_csv("../input/hta-tagging/test.csv")

Test.head(5)
def ReadFileToDataFrame(df,path):

    for i in range(len(df.Filename)):

        Filename = df.Filename[i];

        CurrentFile = "../input/hta-tagging/{2}/{2}/{0}/{1}".format(Filename.split("-")[0],Filename,path);

        File = open(CurrentFile, "r");

        contents = File.read();

        contents = contents.replace("\r\n","\n")

        contents = contents.replace("\r","\n")

        df.Filename[i] = contents

        continue;

    df = df.rename(columns={"Filename":"Text"},inplace=True);

ReadFileToDataFrame(Train,"train-data");

Train.head(5)
import codecs

def ReadTestFileToDataFrame(df,path): # เนื่องจาก Test Set มีไฟล์ Encoding cp1252 อยู่ จึงต้องอ่านด้วยวิธีี้

    df["Text"] = [""]*len(df.Id);

    for i in range(len(df.Id)):

        Filename = df.Id[i];

        CurrentFile = "../input/hta-tagging/{2}/{2}/{0}/{1}".format(Filename.split("-")[0],Filename,path);

        def Do(File):

            contents = File.read();

            contents = contents.replace("\r\n","\n")

            contents = contents.replace("\r","\n")

            df.Text[i] = contents

        try:

            with codecs.open(CurrentFile, encoding='cp1252') as File:

                Do(File);

        except:

            with codecs.open(CurrentFile, encoding='utf-8') as File:

                Do(File);

        continue;

    df = df.rename(columns={"Filename":"Text"},inplace=True);

ReadTestFileToDataFrame(Test,"test-data");

Test.head(5)
from sklearn.preprocessing import OneHotEncoder

def AnsEncoder(df, col_name):

    for i in range(len(df[col_name])):

        val = 0;

        txt = df.loc[i,col_name];

        if (txt == "P"): val = 2;

        if (txt == "Q"): val = 1;

        if (txt == "N"): val = 0;

        df.loc[i,col_name] = val;

        continue;

    df[col_name] = pd.to_numeric(df[col_name]);
AnsEncoder(Train, "Blinding of intervention");

AnsEncoder(Train, "Blinding of Outcome assessment");

Train.head(5)
def remove_ref(text):

    newtext = "";

    for line in text.split("\n"):

        if (line.lower()=="references" or line.lower()=="reference"): break;

        else: newtext += line + "\n";

    return text;



Train.Text = Train.Text.apply(lambda x : remove_ref(x))

Test.Text = Test.Text.apply(lambda x : remove_ref(x))
def FindBeforeColon(text):

    def kwreplace(cutted):

        cutted = cutted.lower();

        if ("key" in cutted): cutted = "key";

        elif ("objective" in cutted): cutted = "objective";

        elif ("reference" in cutted): cutted = "reference";

        elif ("method" in cutted): cutted = "method";

        elif (cutted.endswith("es")): cutted = cutted[:-2];

        elif (cutted.endswith("s")): cutted = cutted[:-1];

        return cutted;

    tmptxt = "";

    curtopic = "";

    topic = [];

    for line in text.split("\n"):

        if (":" in line):

            cutted = line.split(":")[0];

            if (len(cutted)>0):

                if (len(cutted.split(" ")) < 6 and cutted[0] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" and (not "Author" in cutted)):

                    cutted = kwreplace(cutted);

                    if (curtopic != ""):

                        topic.append((curtopic, tmptxt));

                        tmptxt = "";

                    curtopic = cutted;

                    tmptxt = "".join(line.split(":")[1:]);

                    continue;

        if (len(line)>0):

            if (not " " in line and line[0] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):

                if (curtopic != ""):

                        topic.append((curtopic, tmptxt));

                curtopic = kwreplace(line);

                tmptxt = "";

                continue;

        tmptxt += "\n" + line;

    return topic;
TO = [];

i = 0;

for txt in Train.Text:

    TO.append((i,FindBeforeColon(txt)));

    i += 1



TestO = [];

i = 0;

for txt in Test.Text:

    TestO.append((i,FindBeforeColon(txt)));

    i += 1;
SelectedCols = ["Text","method","key"]
def WriteToPD(df,arr):

    arrlength = len(arr);

    PlaceholderList = [None]*df.shape[0];

    for idx, optionlist in arr:

        if (idx % 100 == 0):

            print("{0}/{1} Completed".format(idx,arrlength));

        for option, text in optionlist:

            if option in SelectedCols:

                if not option in df:

                    df[option] = PlaceholderList;

                df.loc[idx,option] = text;
WriteToPD(Train,TO)

Train.head()
WriteToPD(Test,TestO)

Train.head()
from sklearn.feature_extraction.text import TfidfVectorizer;



class TFIDFVector:

    def __init__(this,train,val):

        print("- TF-IDF Vector: Initializing")

        

        this.tfv = TfidfVectorizer(min_df=3,  max_features=None, 

            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',

            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,

            stop_words = 'english');

        

        print("- TF-IDF Vector: Preparing");

        

        # ถ้า Train เป็น Array 2 มิติ

        if type(train[0]) != str:

            tl2 = [];

            for l in train:

                tl2 += [txt for txt in l if txt != None];

            for l in val:

                tl2 += [txt for txt in l if txt != None];

            print("- TF-IDF Vector: Fitting, This might take a while");

            this.tfv.fit(tl2);

            print("- TF-IDF Vector: Fitted");

            

            del(tl2);

        else: # ถ้า Train เป็น Array 1 มิติ

            print("- TF-IDF Vector: Fitting, This might take a while");

            this.tfv.fit(list(train)+list(val));

            print("- TF-IDF Vector: Fitted");

            

        print("- TF-IDF Vector: Initialized")

        

    def CreateVector(this, val, reduceshape = True):

        # สร้าง Vector สำหรับ 1 Column (Train หรือ Validation) ใน Dataframe

        print("- TF-IDF Vector: Creating Vector")

        # ถ้า Train เป็น Array 2 มิติ

        if type(val[0]) != str:

            ValLength = len(val);

            tmplist = []; i = 0;

            

            for ll in val:

                if (i % max(ValLength//10,1) == 0): print("- TF-IDF Vector: Creating Vector {0}/{1} Completed".format(i,ValLength));

                tmplist.append((this.tfv.transform(['' if txt is None else txt for txt in ll])));

                i += 1;

            

            v = np.stack(np.array([x.toarray() for x in tmplist]))

            del(tmplist)

            

            if (reduceshape):

                nsamples = v.shape[0];

                value2 = np.prod(v.shape[1:])

                v = v.reshape((nsamples,value2))

            

            to_return = v;

            

        else: 

            to_return = this.tfv.transform(val)

        

        print("- TF-IDF Vector: Vector Created")

        return to_return;

        
import spacy;

from sklearn.svm import SVC

from sklearn.multioutput import MultiOutputRegressor

class SVMModel:

    def __init__(this):

        return;

    def Train(this,X, Y):

        print("SVMModel: Training Step will be combined with Predict State")

        this.X = X; this.Y = Y;

    def SetVector(this,Vector):

        this.VC = Vector;

    def Predict(this,XVal):

        X = this.VC.CreateVector(this.X); Y = this.Y;

        del(this.X);

        

        this.svc = SVC(random_state=seed)

        

        print("SVMModel: Training")

        this.svc.fit(X, Y); del(X); del(Y);

        print("SVMModel: Finished Training")

        

        print("SVMModel: Predicting");

        valv = this.VC.CreateVector(XVal);

        del(XVal);

        tR = this.svc.predict(valv);

        del(valv);

        print("SVMModel: Predicted");

        

        return tR;
def PredictModel(TX,TY,VX,Model):

    print("Initializing Model")

    model = Model;

    print("Training Model")

    model.Train((TrainX),(TrainY));

    print("Predicting")

    global PredictedY; # Global for Debugging Perpous

    PredictedY = model.Predict(VX);

    return PredictedY.astype(int);
Train
Test
from sklearn.metrics import accuracy_score as ScoreAcc

from sklearn.model_selection import train_test_split as TTS;



TrainX, ValX, TrainY, ValY = TTS(Train[SelectedCols], Train['Blinding of intervention'], test_size=0.2, random_state = seed);

_1, _2, _3, ValY2 = TTS(Train[SelectedCols], Train['Blinding of Outcome assessment'], test_size=0.2, random_state = seed);

# if ((TrainX != _1).any() or (ValX != _2).any()): raise Exception();

TrainX = np.array(TrainX);

TrainY = np.array(TrainY);

ValX = np.array(ValX);

ValY = np.array(ValY);

ValY2 = np.array(ValY2);

TrainY2 = Train['Blinding of Outcome assessment']; TrainY2 = np.array(TrainY2);

TestX = Test[SelectedCols]; TestX = np.array(TestX);



Vector = TFIDFVector(np.array(Train[SelectedCols]), TestX);



ValX = Vector.CreateVector(ValX)



SVM = SVMModel();

SVM.SetVector(Vector);

PredA = PredictModel(TrainX, TrainY , TestX, SVM);

VPredA = SVM.svc.predict(ValX);

AccA = ScoreAcc(ValY,VPredA)



SVM = SVMModel();

SVM.SetVector(Vector);

PredB = PredictModel(TrainX, TrainY2, TestX, SVM);

VPredB = SVM.svc.predict(ValX)

AccB = ScoreAcc(ValY2,VPredB);



def Convert(i):

    if i == 0: return "N";

    if i == 1: return "Q";

    if i == 2: return "P";





FinalPred = [Convert(PredA[i])+Convert(PredB[i]) for i in range(len(PredA))];
Test.Prediction = FinalPred;

SubmissionTest = Test[["Id","Prediction"]];

SubmissionTest.to_csv("submission.csv",index=False)

SubmissionTest.head(5);
def Convert(i):

    if i == 0: return "N";

    if i == 1: return "Q";

    if i == 2: return "P";



FinalPred = [Convert(VPredA[i])+Convert(VPredB[i]) for i in range(len(VPredA))];



CorrectAns = [Convert(ValY[i])+Convert(ValY2[i]) for i in range(len(ValY))];

from sklearn.metrics import accuracy_score as ScoreAcc

FinAcc = ScoreAcc(CorrectAns,FinalPred)*100;

print("A set          = {0:.2f}%".format(AccA))

print("B set          = {0:.2f}%".format(AccB))

print("คะแนนที่คาดว่าจะได้ = {0:.2f}%".format(FinAcc))