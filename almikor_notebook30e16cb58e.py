import numpy as np # linear algebra
a = np.array([[2, 1], [3, 4]])
b = np.linalg.eigh(a)
b
print (b)
from Tkinter import *

# родительский элемент

root = Tk()

# устанавливаем название окна

root.title("Quadratic calculator")

# устанавливаем минимальный размер окна 

root.minsize(325,230)

# выключаем возможность изменять окно

root.resizable(width=False, height=False)

 

# создаем рабочую область

frame = Frame(root)

frame.grid()

 

# поле для ввода первого аргумента уравнения (a)

a = Entry(frame, width=3)

a.grid(row=1,column=1,padx=(10,0))

 

# текст после первого аргумента

a_lab = Label(frame, text="x**2+").grid(row=1,column=2)

 

# поле для ввода второго аргумента уравнения (b)

b = Entry(frame, width=3)

b.grid(row=1,column=3)

# текст после второго аргумента

b_lab = Label(frame, text="x+").grid(row=1, column=4)

 

# поле для ввода третьего аргумента уравнения (с)

c = Entry(frame, width=3)

c.grid(row=1, column=5)

# текст после третьего аргумента

c_lab = Label(frame, text="= 0").grid(row=1, column=6)

 

# кнопка решить

but = Button(frame, text="Solve").grid(row=1, column=7, padx=(10,0))

 

# место для вывода решения уравнения

output = Text(frame, bg="lightblue", font="Arial 12", width=35, height=10)

output.grid(row=2, columnspan=8)

 

# запускаем главное окно

root.mainloop()
