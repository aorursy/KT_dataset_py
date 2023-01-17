import tkinter as tk
window = tk.Tk()
window.mainloop()
window = tk.Tk()
# code about what will happen in the window
window.mainloop()
window = tk.Tk()
label = tk.Label(text='Hello world!')
label.place(x=20, y=20)
window.mainloop()
window = tk.Tk()
entry = tk.Entry()
entry.place(x=20, y=20)
window.mainloop()
window = tk.Tk()
button = tk.Button(text='Click me!')
button.place(x=20, y=20)
window.mainloop()
window = tk.Tk()

def display_message():
    message_label = tk.Label(text='That button was clicked!')
    message_label.place(x=20, y=60)

button = tk.Button(text='Click me!', command=display_message)
button.place(x=20, y=20)
window.mainloop()
window = tk.Tk()

window.geometry('800x600')
window.title('SAMPLE WINDOW')

window.mainloop()
number = int(input())
print(number ** 2)
# import tkinter as tk
# The above line is commented because tkinter has already been imported in the beginning with the nickname tk.

window = tk.Tk() # creates a window.
window.geometry('320x160') # dimensions of the window.
window.title('Square finder') # title of the window.

number_entry = tk.Entry() # entry to get input.
number_entry.place(x=20, y=20) # places entry on screen.

def find_square():
    ''' This function finds the square of the number in the entry and
        displays the output as a label on the window'''
    
    number_in_string_format = number_entry.get() # gets the input in string format.
    integer = int(number_in_string_format) # Converts input to integer. If input is not an integer, will show error.
    
    square = integer ** 2 # finds square of input
    square_in_string_format = str(square) # converts square  to string, so that it can be displayed as a label.
    
    square_label = tk.Label(text=square_in_string_format) # makes a label with the square as text.
    square_label.place(x=200, y=20) # places the label on the window.
    
button = tk.Button(text='Calculate', command=find_square) 
# if this button is clicked, then the above function will be implemented.  

button.place(x=120, y=80) # places the button on the window.

window.mainloop() # finishes the loop of the window, do that the commands in the loop can be implemented.