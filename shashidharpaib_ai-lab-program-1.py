# Create a 3x3 tic tac toe board of "" strings for each value
#board = [' ' for i in range(9)]
# Create a function to display your board
    
def display_board(board):
    print()
    print('   |   |')
    print(' ' + board[0] + ' | ' + board[1] + ' | ' + board[2])
    print('   |   |')
    print('-----------')
    print('   |   |')
    print(' ' + board[3] + ' | ' + board[4] + ' | ' + board[5])
    print('   |   |')
    print('-----------')
    print('   |   |')
    print(' ' + board[6] + ' | ' + board[7] + ' | ' + board[8])
    print('   |   |') 
    print()
    print()
#Create a function to check if anyone won, Use marks "X" or "O"
def check_win(mark, board):
#     print(board)
    
    return ((board[6] == mark and board[7] == mark and board[8] == mark) or # across the top
    (board[3] == mark and board[4] == mark and board[5] == mark) or # across the middle
    (board[0] == mark and board[1] == mark and board[2] == mark) or # across the bottom
    (board[7] == mark and board[4] == mark and board[0] == mark) or # down the middle
    (board[7] == mark and board[4] == mark and board[1] == mark) or # down the middle
    (board[8] == mark and board[5] == mark and board[2] == mark) or
    (board[6] == mark and board[3] == mark and board[0] == mark) or# down the right side
    (board[6] == mark and board[4] == mark and board[2] == mark) or # diagonal
    (board[8] == mark and board[4] == mark and board[0] == mark)) # diagonal

# Create a function to check its a Draw
def check_draw(board):
        return ' ' not in board
# Create a Function that makes a copy of the board
def board_copy(board):
    newBoard = [item for item in board]
    return newBoard
#Immediate move checker
def test_win_move(move, player_mark, board):
    if(board[move] == ' '):
        board[move] = player_mark
        return check_win(player_mark, board)
    else:
        print('Invalid possition')
        return False
    
#Strategy if others fail
def win_strategy(board):
    print(board)
    display_board(board)
    if(board[4]) == ' ':
        return 4
    for i in [0, 2, 6, 8]:
        if(board[i] == ' '):
            return i
    for i in [1, 3, 5, 7]:
        if(board[i] == ' '):
            return i
# Agents move
def get_agent_move(board):
    # Return agent move with your strategy
    bCopy = board_copy(board)
    for i in range(0, 9):
        if(board[i] == ' ' and test_win_move(i, 'O', bCopy)):
            return i
    for i in range(0, 9):
        if(board[i] == ' ' and test_win_move(i, 'X', bCopy)):
            return i



        
    return win_strategy(board)
# Assemble the game
def tictactoe():
    ### Note you need to recreate your board again here if you wish to play the game more than once
    board = [' ' for i in range(9)]
    print('Welcome')
    while(not check_draw(board)):
        display_board(board)
        move = int(input('Your turn! Enter position : '))
        checkWin = test_win_move(move, 'X', board)
        print(checkWin)
        checkDraw = check_draw(board)
        display_board(board)
        if(checkWin):
            print('You won  !')
            display_board(board)
            break
        if(checkDraw):
            print('Draw...')
            display_board(board)
            break
        print('Agent thinking....')
        agentMove = get_agent_move(board)
        checkWin = test_win_move(agentMove, 'O', board)
        checkDraw = check_draw(board)
        if(checkWin):
            print('You lost  !')
            display_board(board)
            break
        if(checkDraw):
            print('Draw...')
            display_board(board)
            break
    
# Play!!!
tictactoe()
class TicTacToe_board:
    def _init_(self):
        self.board = None
        self.mark = None
        self.player_mark = None
        
        
    def display_board(self):
        print('     |     |')
        print('  ' + self.board[0] + '  |  ' + self.board[1] + '  |  ' + self.board[2])
        print('     |     |')
        print('-----------------')
        print('     |     |')
        print('  ' + self.board[3] + '  |  ' + self.board[4] + '  |  ' + self.board[5])
        print('     |     |')
        print('-----------------')
        print('     |     |')
        print('  ' + self.board[6] + '  |  ' + self.board[7] + '  |  ' + self.board[8])
        print('     |     |')
        print('                 ')

    
    
    def board_copy(self, board):
        return [x for x in board]
    
    # Check if a person has won
    def check_win(self, player, board):# Player is X or O
        ## If the player has won then there must be "n" consecutive Player values
        #Check Horizontal
        board = [board[i:i+3] for i in range(0, len(board), 3)]
        horizontal = [player]*3 in board

        #Check Vertical
        vertical = [player]*3 in [list(x) for x in list(zip(*board))]

        #Check Right Diagnol
        left = all(board[i][i] == player for i in range(3))

        #Left Diagnol
        right = all(board[i][2-i] == player for i in range(3))

        return horizontal or vertical or left or right
    
    def check_draw(self):
        return " " not in self.board
    
    def test_win_move(self, move, player_mark, board):
        test_b = self.board_copy(board)
        test_b[move] = player_mark
        return self.check_win(player_mark, test_b)
    
    
    def test_fork_move(self, move, player_mark, board):
        # Determines if a move opens up a fork
        test_b = self.board_copy(board)
        test_b[move] = player_mark
        winning_moves = 0
        for i in range(9):
            if test_b[i] and self.test_win_move(i, player_mark, test_b):
                winning_moves += 1
        return winning_moves >= 2
    
    def final_stategy(self):
        # Play center
        if self.board[4] == ' ':
            return 4
        # Play a corner
        for i in [0, 2, 6, 8]:
            if self.board[i] == ' ':
                return i
        #Play a side
        for i in [1 ,3, 5, 7]:
            if self.board[i] == ' ':
                return i
        
        
    def get_agent_move(self):
        # Check if Agent wins or Sabatoge if Players wins
        for i in range(9):
            if self.board[i] == ' ':
                if self.test_win_move(i, self.mark, self.board):
                    return i
                elif self.test_win_move(i, self.player_mark, self.board):
                    return i
        temp = None
        count = 0
        for i in range(9):
            if self.board[i] == ' ':
                if self.test_fork_move(i, self.mark, self.board):
                    return i
                elif self.test_fork_move(i, self.player_mark, self.board):
                    temp = i
                    count += 1
        if count == 1:
            return temp
        elif count == 2:
            for i in [1, 3, 5, 7]:
                if self.board[i] == ' ':
                    return i
        return self.final_stategy()
        
    # Player plays    
    def player_moves(self):
        self.display_board()
        print("Which spot (0-8)")
        move = int(input()) 
        while self.board[move] != " ":
            self.display_board()
            print("Please pick a valid move (0-8)")
            move = int(input())
        self.board[move] = self.player_mark
        self.display_board()

    #Agent plays
    def agent_moves(self):
        move = self.get_agent_move()
        self.board[move] = self.mark
        self.display_board()
    
    # Assemble the game
    def tictactoe(self):
        Playing = True
        while Playing:
            self.board = [" " for i in range(9)]
            # Choose Marks
            print('X or O')
            self.player_mark = input()
            self.mark = 'X' if self.player_mark == "O" else "O"

            #Choose start
            print("Want to go first [y,n]")
            flag = '1' if input() == 'y' else '0'

            Ingame = True
            while Ingame:

                if flag == '1':
                    self.player_moves()
                    mark = self.player_mark
                else:
                    self.agent_moves()
                    mark = self.mark
                
                #Check if someone won
                if self.check_win(mark, self.board):
                    Ingame = False
                    if mark == self.player_mark:
                        print("Player wins!!!")
                    else:
                        print("Agent wins")
                    break
                
                # Check if game Draws
                if self.check_draw():
                    print('Draw!!!')
                    break
                
                #Switch to comp or person moves
                flag = '0' if flag == '1' else '1'
            
            # Another game 
            print("Another game [y,n]?")
            if input() == 'n':
                Playing = False
        print('Thank you for playing!')

b = TicTacToe_board()
b.tictactoe()