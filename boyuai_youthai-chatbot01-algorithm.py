class Node:  

    def __init__(self,val = None,left = None,right = None):  

         self.val = val

         self.left = left    

         self.right = right 
F=Node('F',None,None)

C=Node('C',None,None)

A=Node('A',None,None)

G=Node('G',F,None)

B=Node('B',A,C)

E=Node('E',None,G)

root=Node('D',B,E)
def preTraversal(root):  

    #前序遍历

    if not root:

        return [] 

    return  [root.val] + preTraversal(root.left) + preTraversal(root.right)



def midTraversal(root): 

    #中序遍历

    if not root:

        return [] 

    return midTraversal(root.left) + [root.val] + midTraversal(root.right)

def afterTraversal(root):  

    #后序遍历

    if not root:

        return [] 

    return  afterTraversal(root.left) + afterTraversal(root.right) + [root.val]

print('前序遍历：')

print(preTraversal(root))

print('中序遍历：')

print(midTraversal(root))

print('后序遍历：')

print(afterTraversal(root))
def inorderTraversal( root): ## 中序遍历

    stack = []

    sol = []

    curr = root

    while stack or curr:

        if curr:

            stack.append(curr)

            curr = curr.left

        else:

            curr = stack.pop()

            sol.append(curr.val)

            curr = curr.right

    return sol
print(inorderTraversal(root))
def preorderTraversal(root):  ## 前序遍历

    stack = []

    sol = []

    curr = root

    while stack or curr:

        if curr:

            sol.append(curr.val)

            stack.append(curr.right)

            curr = curr.left

        else:

            curr = stack.pop()

    return sol



def postorderTraversal( root): ## 后序遍历

    stack = []

    sol = []

    curr = root

    while stack or curr:

        if curr:

            sol.append(curr.val)

            stack.append(curr.left)

            curr = curr.right

        else:

            curr = stack.pop()

    return sol[::-1]
print(preorderTraversal(root))

print(postorderTraversal(root))