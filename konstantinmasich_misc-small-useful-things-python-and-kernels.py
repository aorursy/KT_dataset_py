def show_bar(current, total):
    """
    Displays progress bar. Use it in loops.
    
    Parameters:
    ----------
    current : int, number of current iteration, starts from 1!
    
    full : int, total amount of iterations
    """
    output = '['+'■'*current+' '*(total-current)+']  '+str(int((current/total)*100))+'%'
    print(output, end="\r")
from time import sleep

def stuff(x=None):
    """Does lots of really important stuff!"""
    sleep(0.5) # A delay to simulate execution of that very improtant stuff
n_loops = 10
for i in range(n_loops):
    stuff()                 # Do useful stuff here
    show_bar(i+1, n_loops)  # Show progress bar
word_list = 'Apple orange tomato banana '.split()
word_list
for word in word_list:
    stuff(word)                                        # Do useful stuff here
    show_bar(word_list.index(word)+1, len(word_list))  # Show progress bar
word_list = str('a '*5).split()
word_list
for word in word_list:
    stuff(word)                                        # Do useful stuff here
    show_bar(word_list.index(word)+1, len(word_list))  # Show progress bar
iteration = 1
for word in word_list:
    stuff(word)                          # Do useful stuff here
    show_bar(iteration, len(word_list))  # Show progress bar
    iteration += 1                       # Don't forget to increment iteration counter
from sklearn.neighbors import KNeighborsRegressor
param_list = ['auto', 'ball_tree', 'kd_tree', 'brute']

for param_val in param_list:
    # Create KNN with 'algorhitm' parameter successively set to one
    # of its values: 'auto', 'ball_tree', 'kd_tree' and 'brute'
    model = KNeighborsRegressor(algorithm=param_val)
    # Fit the model.. Evaluate.. Store result.. Do what you need.. Etc..
    sleep(0.5)
    show_bar(param_list.index(param_val)+1, len(param_list))
HORIZONTAL_LINE = '-'*80 + '\n'
# Or maybe you can call it HL for short:
HL = '-'*80 + '\n'
# Or HR like in HTML tag <hr>... Do as you like!
print(HL + 'Experiment 1\n\t• Mass 1 = 12.3481 kg\n\t• Mass 2 = 42.0011 kg\n')
print(HL + 'Experiment 2\n\t• Mass 1 = 11.2894 kg\n\t• Mass 2 = 48.4028 kg')
HL = '\n' + '='*30 + '\n'
print(HL + 'A nu cheeki breeki i v damki' + HL)
# Something like:
class color:
    RED   = "\033[1;31m"  
    BLUE  = "\033[1;34m"
    CYAN  = "\033[1;36m"
    GREEN = "\033[0;32m"
    BOLD  = "\033[;1m"
    END   = '\033[0m'
print(color.RED + 'Red text, ' + color.BLUE + 'blue text' + color.END)
print('Experiment results:')
print('\t - Parameter 1 = 347\n\t - Parameter 2 = 7234\n\t - Parameter 3 = 234')
print('\t ......')
print('\t - Parameter n = 193')
print(color.RED  + '\t - Total delta = -123.12')
print(color.CYAN + '\t - Resulting sum = 939248.34' + color.END)
print('The beauty is in the eye of the ' + color.BOLD + 'beer holder ' + color.END + '🍺')
import matplotlib.pyplot as plt

def print_latex(text, size=15):
    a = r'\' + text + r'
    ax = plt.axes([0,0,0.001,0.001]) #left,bottom,width,height
    ax.set_xticks([])
    ax.set_yticks([])
    plt.text(0.3, 0.4, text, size=size)
    plt.show()
formula = '$y = X\\beta + \\varepsilon$'
print_latex(formula, size=20)
formula = '$Q_c(\\mu, X^L) = \mathrm{E}_n\\nu_n^k=\\frac{1}{N}\\sum_{N=1}^n\\nu_n^k$'
print_latex(formula, size=20)
formula  = '$1st: y = ax + b$ \n'
formula += '$2nd: y = ax^2 + bx + c$ \n'
formula += '$3rd: y = ax^3 + bx^2 + cx + d$'
print_latex(formula, size=20)