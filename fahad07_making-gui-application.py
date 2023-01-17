import tkinter as tk
from tkinter import *
from tkinter.ttk import Combobox
window = tk.Tk()
window.geometry("500x500")
window.title("My Tkinter")
def aly():
    gender = 'm'
    if(v0.get()==1):
        gender='m'
    else:
        gender='f'
    lblr.config(text=gender)
    
#label, button, entry
lblr = Label(window, text="Result")
lblr.grid(column=0, row=6)
v0 = IntVar()
v0.set(1)
r1 = Radiobutton(window, text="male", variable=v0, value=1)
r1.grid(column=0, row=1)
r2 = Radiobutton(window, text="fe-male", variable=v0, value=2)
r2.grid(column=1, row=1)

v1 = IntVar()
v2 = IntVar()
c1 = Checkbutton(window, text="Cricket", variable=v1)
c1.grid(column=1, row=2)
c2 = Checkbutton(window, text="Tennis", variable=v2)
c2.grid(column=0, row=2)

data = ("Lucknow", "Kanpur", "Varanasi", "Allahabad")
cb = Combobox(window, values=data)
cb.grid(column=0, row=3)

