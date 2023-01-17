!pip install binarytree

#!pip install anytree





from binarytree import Node



root = Node(1)

root.left = Node(2)

root.right = Node(3)

root.left.left = Node(4)

root.left.right = Node(5)

root.right.left = Node(6)

root.right.right = Node(7)

root.right.left.left = Node(8)

root.right.left.right = Node(9)

root.right.right.right = Node(10)

                             

print("Root height: ", root.height)

print("Inorder: ", root.inorder)

print("Preorder: ", root.preorder)

print("Postorder: ", root.postorder)

print(root)




