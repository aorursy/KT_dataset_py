#################################################################list of words
words = ['What', 'wonderful', 'world', "backend", "exercises", 'logo', 'Python', 'Pascal', "asthma", "happy", "father",
         "affection", "enthusiasm", "marvelous", "encyclopedia", "environmental", "strength", "Weakness", "Love",
         "funny"
    , "pretty", "ugly", "zinc", 'monitor', 'foolish', 'Corona', 'cloud', 'Pubg', 'classic', 'counter', 'strike',
         'breathe', 'width'
    , 'style', 'lawsuit', 'lawyer', 'doctor', 'fashion', 'icon', 'metabolism', 'physical', 'elements', 'compound',
         'heaven', 'hell',
         'smile', 'worry', 'dumbo', 'xmas', 'axis', 'yatch', 'watch', 'quite', 'quit', 'avengers', 'actor', 'GOT',
         'FRIENDS', 'excursion', 'conjuring', 'annabelle', 'symptom', 'virus', 'excel', 'college', 'school', 'state',
         'family', 'mother', 'father', 'brother', 'sister', 'frame']


##########################################################FUNCTIONS
def labelslider():
    global c, sliderword
    text = ('''------SPEED UP----SPEED UP----SPEED UP----SPEED UP-------''')
    if (c >= len(text)):
        c = 0
        sliderword = ""
    sliderword += text[c]
    c += 1
    fontlabel.configure(text=sliderword)
    fontlabel.after(30, labelslider)


def startgame(event):
    global score, miss
    if (timeleft == 60):
        times()
    if (wordentry.get() == wordlabel['text']):
        score += 1
        scorecount.configure(text=score)
    else:
        miss += 1
        misscount.configure(text=miss)
    random.shuffle(words)
    wordlabel.configure(text=words[0])
    wordentry.delete(0, 'end')


def times():
    global timeleft, score, miss
    if (timeleft >= 11):
        pass
    else:
        timecount.configure(fg="orange")
    if (timeleft > 0):
        timeleft -= 1
        timecount.configure(text=timeleft)
        timecount.after(1000, times)
    else:
        message.configure(text=(f"RESULT = {score} words per minute"), fg="black")
        r = messagebox.askretrycancel('notification', "DO YOU WANT TO PLAY IT AGAIN")
        if (r == TRUE):

            rr = messagebox.askyesno("Start Game", "START GAME")
            if (rr == TRUE):
                score = 0
                miss = 0
                timeleft = 60
                wordentry.delete(0, 'end')
                timecount.configure(text=timeleft, fg="black")
                scorecount.configure(text=score)
                misscount.configure(text=miss)
                wordlabel.configure(text=words[0])
                message.configure(text="TYPE GIVEN WORD QUICKLY AND HIT 'ENTER'", fg="black")
            else:
                exit()

        else:
            exit()


from tkinter import *
import random
from time import sleep
from tkinter import messagebox
import threading

###############################################################VARIABLE

c = 0
score = 0
miss = 0
timeleft = 60
sliderword = ""

################################################################ROOT
root = Tk()
root.geometry('885x500+330+100')
root.configure(bg="teal")
root.title("TYPING MASTER")
# root.iconbitmap(r"c:\Users\91955\Desktop\saumya_code\images-1.ico")
# root.iconbitmap('ss.ico')

#####################################################################LABELS
fontlabel = Label(root, text='', font=('times', '25', 'bold'), bg="teal",
                  fg='maroon', justify="center")
fontlabel.place(x=-8, y=10)
labelslider()

random.shuffle(words)
wordlabel = Label(root, text=words[0], font=('times', '30', 'bold'), bg="teal",
                  fg='midnight blue', justify="center")
wordlabel.place(x=355, y=250)
scorelabel = Label(root, text="CORRECT", font=('times', '24', 'bold'), bg="teal",
                   fg='maroon', justify="center")
scorelabel.place(x=50, y=160)
scorecount = Label(root, text=score, font=('times', '20', 'bold'), bg="teal",
                   fg='black', justify="center")
scorecount.place(x=100, y=200)
misslabel = Label(root, text="WRONG", font=('times', '24', 'bold'), bg="teal",
                  fg='maroon', justify="center")
misslabel.place(x=50, y=260)
misscount = Label(root, text=miss, font=('times', '20', 'bold'), bg="teal",
                  fg='black', justify="center")
misscount.place(x=90, y=300)
timelabel = Label(root, text="TIME LEFT", font=('times', '24', 'bold'), bg="teal",
                  fg='maroon', justify="center")
timelabel.place(x=660, y=160)
timecount = Label(root, text=timeleft, font=('times', '20', 'bold'), bg="teal",
                  fg='BLACK', justify="center")
timecount.place(x=720, y=200)
message = Label(root, text="TYPE GIVEN WORD QUICKLY AND HIT 'ENTER' ", font=('times', '25', 'bold'), bg="teal",
                fg="BLACK", justify="center")
message.place(x=40, y=400)

#########################################################################entry

wordentry = Entry(root, font=('times', '20', 'bold'), bg="gray", bd="10",
                  fg='black', justify="center", width=25)
wordentry.place(x=250, y=300)
wordentry.focus_set()
root.bind('<Return>', startgame)
root.mainloop()

