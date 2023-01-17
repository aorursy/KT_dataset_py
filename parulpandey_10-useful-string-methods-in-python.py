from IPython.core.interactiveshell import InteractiveShell  

InteractiveShell.ast_node_interactivity = "all"
sentence = 'algorithm'

sentence.center(15,'#')
sentence = 'She sells seashells by the seashore. The shells she sells are surely seashells'

sentence.count('seashells')

sentence.count('seashells',9,25)
sentence = 'She sells seashells by the seashore. The shells she sells are surely seashells'

sentence.find('seashells')

sentence.find('seashells',0,9)

sentence.find('s',5,10)

sentence.rfind('seashells')
sentence = 'Queue IS another FUNDAMENTAL data STRucture AND IS a close COUSIN of the STACK'

sentence.swapcase()
#string.startswith()



sentence = 'Binary Search is a classic recursive algorithm'

sentence.startswith("Binary")

sentence.startswith("Search",7,20)
#string.split()



fruits = 'apples, mangoes, bananas, grapes'

fruits.split()

fruits.split(",",maxsplit = 2)
#string.rsplit()

fruits.rsplit(",",maxsplit = 1)
"san francisco".capitalize()
"san francisco".upper()
"san francisco".title()
#str.rjust

text = 'Binary Search'

print(text.rjust(25),"is a classic recursive algorithm")
#str.ljust

text = 'Binary Search'

print(text.ljust(25),"is a classic recursive algorithm")
#str.strip

string = '#.......Section 3.2.1 Issue #32......'

string.strip('.#!')
#str.rstrip

string.rstrip('.#!')

string.lstrip('.#!')
'7'.zfill(3)

'-21'.zfill(5)

'Python'.zfill(10)

'Python'.zfill(3)