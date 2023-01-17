!pip install binarytree



from binarytree import Node



print("Tree 1")

root = Node(30)

print(root)

print("Tree 2")

root.right = Node(40)

print(root)

print("Tree 3")

root.left = Node(24)

print(root)

print("Tree 4")

root.right.right = Node(58)

print(root)

print("Tree 5")

root.right.right.left = Node(48)

print(root)

print("Tree 6")

root.left.right = Node(26)

print(root)

print("Tree 7")

root.left.right = Node(25)

print(root)
print("1")

root = Node(1)

root.right = Node(2)

root.right.right = Node(3)

print(root)



print("2")

root = Node(1)

root.right = Node(3)

root.right.left = Node(2)

print(root)



print("3")

root = Node(2)

root.left = Node(1)

root.right = Node(3)

print(root)



print("4")

root = Node(3)

root.left = Node(1)

root.left.right = Node(2)

print(root)



print("5")

root = Node(3)

root.left = Node(2)

root.left.left = Node(1)

print(root)
root = Node(62)

root.left = Node(44)

root.right = Node(78)

root.right.right = Node(88)

root.left.left = Node(17)

root.left.right = Node(50)

root.left.right.left = Node(48)

root.left.right.right = Node(54)

print(root)
!pip install anytree

from anytree import Node, RenderTree

node = Node("4,8,12")

leftleft = Node("1,2,3", parent=node)

left = Node("5,6,7", parent=node)

right = Node("9,10,11", parent=node)

rightright = Node("13,14,15", parent=node)





for pre, fill, node in RenderTree(node):

    print("%s%s" % (pre, node.name))
node = Node("5")

for pre, fill, node in RenderTree(node):

    print("%s%s" % (pre, node.name))

print("insert")

node = Node("5,16")

for pre, fill, node in RenderTree(node):

    print("%s%s" % (pre, node.name))

print("insert")

node = Node("5,16,22")

for pre, fill, node in RenderTree(node):

    print("%s%s" % (pre, node.name))

print("Overflow")



node = Node("22")

for pre, fill, node in RenderTree(node):

    print("%s%s" % (pre, node.name))

print("insert")



node = Node("22")

left = Node("5,16", parent=node)

right = Node("45", parent=node)

for pre, fill, node in RenderTree(node):

    print("%s%s" % (pre, node.name))

print("insert")



node = Node("22")

left = Node("2,5,16", parent=node)

right = Node("45", parent=node)

for pre, fill, node in RenderTree(node):

    print("%s%s" % (pre, node.name))

print("insert")



node = Node("10,22")

left = Node("2,5", parent=node)

mid = Node("16", parent=node)

right = Node("45", parent=node)

for pre, fill, node in RenderTree(node):

    print("%s%s" % (pre, node.name))

print("insert")

    

    

node = Node("10,22")

left = Node("1,2,5", parent=node)

mid = Node("12,16,18", parent=node)

right = Node("30,45,50", parent=node)

for pre, fill, node in RenderTree(node):

    print("%s%s" % (pre, node.name))

from binarytree import Node





root = Node(10)

print(root)

print("-----------")



root = Node(1)

root.right = Node(16)

print(root)

print("-----------")

root = Node(10)

root.right = Node(16)

print(root)

print("-----------")

root = Node(16)

root.left = Node(10)

print(root)

print("-----------")

root = Node(16)

root.left = Node(10)

root.left.right = Node(12)

print(root)

print("-----------")

root = Node(12)

root.left = Node(10)

root.right = Node(16)

print(root)

print("-----------")

root = Node(12)

root.left = Node(10)

root.right = Node(16)

root.right.left = Node(14)

print(root)

print("-----------")

root = Node(14)

root.left = Node(12)

root.right = Node(16)

root.left.left = Node(10)

print(root)

print("-----------")

root = Node(14)

root.left = Node(12)

root.right = Node(16)

root.left.left = Node(10)

root.left.right = Node(13)

print(root)

print("-----------")

root = Node(13)

root.left = Node(12)

root.right = Node(14)

root.left.left = Node(10)

root.right.right = Node(16)

print(root)

print("-----------")

root = Node(13)

root.left = Node(12)

root.right = Node(14)

root.left.left = Node(10)

root.right.right = Node(16)

root.right.right.left = Node(15)

print(root)

print("-----------")

#Final

root = Node(15)

root.left = Node(13)

root.right = Node(16)

root.left.left = Node(12)

root.left.right = Node(14)

root.left.left.left = Node(10)

print(root)

print("-----------")