def iterative_counter(initial_value):
    num = initial_value
    # 1. The condition
    while (num < 5):
        
        # 2 The operation
        print(num, end=" ")
        
        # 3. The increment
        num += 1
    
def recursive_counter(num):
    # 1. The base case
    if (num >= 5):
        return
    
    # 2. The operation
    print(num, end=" ")
    
    # 3. The recursive call
    recursive_counter(num + 1)

print("Iteration: ")
iterative_counter(0)

print("\nRecursion: ")
recursive_counter(0)
class BinaryTreeNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
        
binary_tree = BinaryTreeNode(1, 
                             BinaryTreeNode(2, BinaryTreeNode(3), BinaryTreeNode(4)), 
                             BinaryTreeNode(5))
def binary_tree_preorder_traversal(node):
    # 1. The base case
    if node is None:
        return
    
    # 2. The operation
    print(node.value, end=" ")
    
    #3. The recursive call(s)
    binary_tree_preorder_traversal(node.left)
    binary_tree_preorder_traversal(node.right)

binary_tree_preorder_traversal(binary_tree)
def binary_tree_inorder_traversal(node):
    # 1. The base case
    if node is None:
        return
    
    # 3. The recursive call part 1
    binary_tree_inorder_traversal(node.left)
    
    # 2. The operation
    print(node.value, end=" ")
    
    # 3. The recursive call part 2
    binary_tree_inorder_traversal(node.right)

binary_tree_inorder_traversal(binary_tree)
def binary_tree_postorder_traversal(node):
    # Implement here
    return
    
binary_tree_inorder_traversal(binary_tree)
def add_one_binary_tree(node):
    # Implement here
    return None
    
    
# I will make a copy of the binary tree for your method to mutate
import copy
new_binary_tree = copy.deepcopy(binary_tree)
add_one_binary_tree(new_binary_tree)

# Print an answer
binary_tree_preorder_traversal(new_binary_tree)
def depth_first_search(node, search_value):
    # 1. The base case
    if node is None:
        return None;
    
    # 2. The operation
    if node.value == search_value:
        return node
    
    # 3. The recursive call(s)
    left = depth_first_search(node.left, search_value)
    if left is not None:
        return left
    
    right = depth_first_search(node.right, search_value)
    if right is not None:
        return right
def leftmost_value(node):
    # 1. Base Case

    # 2. Operation - If I can't go left anymore, I am the leftmost.
    if node.left is None:
        return node.value;
    
    # 3. Recursive Case
    

print(leftmost_value(binary_tree))
def count_nodes(node):
    # 1. Base case
    if node is None:
        return 0
    
    # 3. Recursive calls
    left_node_count = count_nodes(node.left)
    right_node_count = count_nodes(node.right)
    
    #2. The operation (Post-Order)
    subtree_node_count = left_node_count + right_node_count + 1
    return subtree_node_count

count_nodes(binary_tree)
def sum_nodes(node):
    # 1. Base case
    if node is None:
        return 0

    # Implement here
    

print(sum_nodes(binary_tree))
def find_max_value(node):
    # 1. Base case
    if node is None:
        return 0
    
    # 3 Recursive calls
    left_max = find_max_value(node.left)
    right_max = find_max_value(node.right)
    
    # 2. Operation (post-order)
    my_max = node.value
    if left_max > my_max:
        my_max = left_max
    if right_max > my_max:
        my_max = right_max
        
    return my_max
    
print("For a positive tree, I found:")
print(find_max_value(binary_tree),)

negative_binary_tree = BinaryTreeNode(-2, BinaryTreeNode(-3), BinaryTreeNode(-4))
print("For a negative tree, I found:")
print(str(find_max_value(negative_binary_tree)) + " (expected: -2)")

