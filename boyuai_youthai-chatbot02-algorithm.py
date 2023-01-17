class Node:  

    def __init__(self,val = None,left = None,right = None):  

         self.val = val

         self.left = left    

         self.right = right
numbers=[2, 5, 8, 3, 1, 4, 9, 0, 7]
def insertNode( data, btnode):

        if data < btnode.val:

            if btnode.left == None:

                btnode.left = Node(data, None, None)

                return

            insertNode(data, btnode.left)

        elif data > btnode.val:

            if btnode.right == None:

                btnode.right = Node(data, None, None)

                return

            insertNode(data, btnode.right)

def midTraversal(root): 

    #中序遍历

    if not root:

        return [] 

    return midTraversal(root.left) + [root.val] + midTraversal(root.right)
if numbers == []:

    root = None

else:

    root = Node(numbers[0],None,None)  

    for i in numbers[1:]:

         insertNode(i,root)
print(midTraversal(root))