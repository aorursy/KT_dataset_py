import random
import re

def getpatterns(symbol):
    return "$$$.{6}|.{3}$$$.{3}|.{6}$$$|$.{2}$.{2}$.{2}|.$.{2}$.{2}$.|.{2}$.{2}$.{2}$|$.{3}$.{3}$|.{2}$.$.$.{2}".replace("$",symbol)

def checkPatterns(pattern,TicTecBoard):
    return len(re.findall(pattern,"".join(TicTecBoard)))

def printTicTecBoard(TicTecBoard):
    print("\n")
    strTicTecBoard=""
    strTicTecBoardLearn=""
    for i in range(0,9):
        strTicTecBoardLearn+= 3*" " +  str(i+1) + " "*3+"|"
        strTicTecBoard+= 3*" " + TicTecBoard[i] + " "*3+"|"
        if (i+1)%3==0:
            strTicTecBoardLearn=strTicTecBoardLearn[0:-1]+" "*25+strTicTecBoard[0:-1] +"\n"+"_"*25+" "*25+"_"*20+"\n"
            strTicTecBoard=""
    
    print(strTicTecBoardLearn[0:-75],"\n")

def getValidPlace(TicTecBoard):
    return [str(i+1) for i,x in enumerate(TicTecBoard) if x=="-"]

patternO=getpatterns("O")
patternX=getpatterns("X")
TicTecBoard=["-" for i in range(9)]
player="O"
try:



    while True:    
        ValidIndexList=getValidPlace(TicTecBoard)
        if player=="O":
            while True:  
                printTicTecBoard(TicTecBoard)
                index=input(" \n Enter Cell Number from Valid Index List  "+str(ValidIndexList )+" : \n")

                if index in ValidIndexList:
                    index=int(index)-1
                    break
                else:
                    print("Plz Enter Valied Place") 

        else:
            machineWin=0
            for place in ValidIndexList:
                testTicTecBoard=list(TicTecBoard)
                testIndex=int(place)-1
                testTicTecBoard[testIndex]=player
                if checkPatterns(patternX,testTicTecBoard)>0: 
                    index=testIndex
                    machineWin=1
            if machineWin==0:
                for place in ValidIndexList:
                    testTicTecBoard=list(TicTecBoard)
                    testIndex=int(place)-1
                    testTicTecBoard[testIndex]="O"
                    if checkPatterns(patternO,testTicTecBoard)>0: 
                        index=testIndex
                        machineWin=-1
            if machineWin==0:
                index=4 if "5" in ValidIndexList else int(random.choice(ValidIndexList))-1



        TicTecBoard[index]=player


        if checkPatterns(getpatterns(player),TicTecBoard)>0: 
            printTicTecBoard(TicTecBoard)
            print('\x1b[6;30;42m'  +player+" is Win "+ '\x1b[0m')  
            break
        elif checkPatterns("-",TicTecBoard)==0: 
            printTicTecBoard(TicTecBoard)
            print("\033[93m Game is Draw  \033[0m")
            break

        else:
            player="X" if player=="O" else "O"



except:
    print("Plz Enter Valied Index")




