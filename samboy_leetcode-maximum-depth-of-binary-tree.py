# 方法1：使用递归
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        # recursion terminator
        if not root:
            return 0 

        # Drill to next level
        left_height = self.maxDepth(root.left) 
        right_height = self.maxDepth(root.right) 
        
        # process
        height = max(left_height, right_height) + 1
        
        return height

Solution().maxDepth(tree.root)
test = [3,9,20,None,None,15,7]
tree = Tree()
tree.construct_tree(test)

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

#####################################################
# https://www.cnblogs.com/qiaojushuang/p/7724936.html
from collections import deque

class Tree(object):
    def __init__(self):
        self.root = None

    def construct_tree(self, values=None):
        if not values:
            return None
        self.root = TreeNode(values[0])
        queue = deque([self.root])
        leng = len(values)
        nums = 1
        while nums < leng:
            node = queue.popleft()
            if node:
                node.left = TreeNode(values[nums]) if values[nums] else None
                queue.append(node.left)
                if nums + 1 < leng:
                    node.right = TreeNode(values[nums+1]) if values[nums+1] else None
                    queue.append(node.right)
                    nums += 1
                nums += 1

    def bfs(self):
        ret = []
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            if node:
                ret.append(node.val)
                queue.append(node.left)
                queue.append(node.right)
        return ret

    def pre_traversal(self):
        ret = []

        def traversal(head):
            if not head:
                return
            ret.append(head.val)
            traversal(head.left)
            traversal(head.right)
        traversal(self.root)
        return ret

    def in_traversal(self):
        ret = []

        def traversal(head):
            if not head:
                return
            traversal(head.left)
            ret.append(head.val)
            traversal(head.right)

        traversal(self.root)
        return ret

    def post_traversal(self):
        ret = []

        def traversal(head):
            if not head:
                return
            traversal(head.left)
            traversal(head.right)
            ret.append(head.val)

        traversal(self.root)
        return ret

# t = Tree()
# t.construct_tree([1, 2, None, 4, 3, None, 5])
# print (t.bfs())
# print (t.pre_traversal())
# print (t.in_traversal())
# print (t.post_traversal())
