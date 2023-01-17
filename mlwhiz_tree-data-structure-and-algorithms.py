class node:

    def __init__(self,val):

        self.val = val

        self.left = None

        self.right = None
root = node(1)

root.left = node(2)

root.right = node(3)
root
def inorder(node):

    if node:

        inorder(node.left)

        print(node.val)

        inorder(node.right)
inorder(root)
def create_bst(array,min_index,max_index):

    if max_index<min_index:

        return None

    mid = int((min_index+max_index)/2)

    root = node(array[mid])

    leftbst = create_bst(array,min_index,mid-1)

    rightbst = create_bst(array,mid+1,max_index)

    root.left = leftbst

    root.right = rightbst

    return root

a = [2,4,5,6,7]

root = create_bst(a,0,len(a)-1)
inorder(root)
def isValidBST(node, minval, maxval):

    if node:

        # Base case

        if node.val<=minval or node.val>=maxval:

            return False

        # Check the subtrees changing the min and max values

        return isValidBST(node.left,minval,node.val) &    isValidBST(node.right,node.val,maxval)

    return True

isValidBST(root,-float('inf'),float('inf'))