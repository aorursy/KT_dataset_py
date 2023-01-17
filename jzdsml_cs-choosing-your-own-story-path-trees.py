class TreeNode:

  def __init__(self, value):

    self.value = value

    self.children = []



  def __repr__(self, level=0):

    # HELPER METHOD TO PRINT TREE!

    ret = "--->" * level + repr(self.value) + "\n"

    for child in self.children:

      ret += child.__repr__(level+1)

    return ret



  def add_child(self, child_node):

    self.children.append(child_node) 



### TEST CODE TO PRINT TREE

company = [

  "Monkey Business CEO", 

  "VP of Bananas", 

  "VP of Lazing Around", 

  "Associate Chimp", 

  "Chief Bonobo", "Produce Manager", "Tire Swing R & D"]

root = TreeNode(company.pop(0))

for count in range(2):

  child = TreeNode(company.pop(0))

  root.add_child(child)



root.children[0].add_child(TreeNode(company.pop(0)))

root.children[0].add_child(TreeNode(company.pop(0)))

root.children[1].add_child(TreeNode(company.pop(0)))

root.children[1].add_child(TreeNode(company.pop(0)))

print("MONKEY BUSINESS, LLC.")

print("=====================")

print(root)

#This program requires keyboard input, which Kaggle doesn't seem to fully support as of now (2019-05-12).

#This program runs well on local computers.





#We will write Wilderness Escape, an online Choose-Your-Own-Adventure. 

#Our users get a unique story experience by picking the next chapter of their adventure. 

#We use the tree data structure to keep track of the different paths a user may choose.



print("Once upon a time...")



######

# TREENODE CLASS

######

class TreeNode:

  def __init__(self, story_piece):

    self.story_piece = story_piece

    self.choices = []

    

  def add_child(self, node):

    self.choices.append(node)

    

  def traverse(self):

    story_node = self

    print(story_node.story_piece)

    while len(story_node.choices) != 0:

      user_choice = int(input("Enter a valid choice to continue the story:"))

      if user_choice in range(1, len(story_node.choices)+1):

        chosen_child = story_node.choices[user_choice - 1]

        print(chosen_child.story_piece)

        story_node = chosen_child

      else:

        print("Invalid choice!!!")

      

      



    

    

######

# VARIABLES FOR TREE

######

#user_choice = input("What is your name?")

#print(user_choice)

story_root = TreeNode("""

You are in a forest clearing. There is a path to the left.

A bear emerges from the trees and roars!

Do you: 

1 ) Roar back!

2 ) Run to the left...

""")



choice_a = TreeNode("""

The bear is startled and runs away.

Do you:

1 ) Shout 'Sorry bear!'

2 ) Yell 'Hooray!'

""")



choice_b = TreeNode("""

You come across a clearing full of flowers. 

The bear follows you and asks 'what gives?'

Do you:

1 ) Gasp 'A talking bear!'

2 ) Explain that the bear scared you.

""")



story_root.add_child(choice_a)

story_root.add_child(choice_b)



choice_a_1 = TreeNode("""

The bear returns and tells you it's been a rough week. After making peace with

a talking bear, he shows you the way out of the forest.



YOU HAVE ESCAPED THE WILDERNESS.

""")



choice_a_2 = TreeNode("""

The bear returns and tells you that bullying is not okay before leaving you alone

in the wilderness.



YOU REMAIN LOST.

""")



choice_a.add_child(choice_a_1)

choice_a.add_child(choice_a_2)



choice_b_1 = TreeNode("""

The bear is unamused. After smelling the flowers, it turns around and leaves you alone.



YOU REMAIN LOST.

""")



choice_b_2 = TreeNode("""

The bear understands and apologizes for startling you. Your new friend shows you a 

path leading out of the forest.



YOU HAVE ESCAPED THE WILDERNESS.

""")



choice_b.add_child(choice_b_1)

choice_b.add_child(choice_b_2)







######

# TESTING AREA

######

story_root.traverse()


