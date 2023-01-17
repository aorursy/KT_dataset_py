# Create a 3x3 tic tac toe board of "" strings for each value
board = [" "] * 9
# Create a function to display your board
def display_board(board):
    print(" %c | %c | %c " % (board[0],board[1],board[2]))    
    print("___|___|___")    
    print(" %c | %c | %c " % (board[3],board[4],board[5]))    
    print("___|___|___")    
    print(" %c | %c | %c " % (board[6],board[7],board[8]))    
    print("   |   |   ") 
    

#Create a function to check if anyone won, Use marks "X" or "O"
def check_win(player_mark, board):
    return (
        (board[0] == player_mark and board[1]== player_mark and board[2]==player_mark)or
        (board[3] == player_mark and board[4]== player_mark and board[5]==player_mark)or
        (board[6] == player_mark and board[7]== player_mark and board[8]==player_mark)or
        (board[1] == player_mark and board[4]== player_mark and board[7]==player_mark)or
        (board[0] == player_mark and board[3]== player_mark and board[6]==player_mark)or
        (board[2] == player_mark and board[5]== player_mark and board[8]==player_mark)or
        (board[0] == player_mark and board[4]== player_mark and board[8]==player_mark)or
        (board[2] == player_mark and board[4]== player_mark and board[6]==player_mark)
    )

# Create a function to check its a Draw
def check_draw(board):
        return " " not in board
    

# Create a Function that makes a copy of the board
def board_copy(board):
    board1=[]
    for i in board:
        board1.append(i)
        
    return board1

#Immediate move checker
def test_win_move(board, player_mark, move):
    bcopy = board_copy(board)
    bcopy[move] = player_mark
    return check_win(player_mark, bcopy)
    
#Strategy if others fail
def win_strategy(board):
    if board[4] == " ":
        return 4
    for i in [0,2,6,8]:
        if board[i] == " ":
            return i
    for i in [1,3,5,7]:
        if board[i] == " ":
            return i
    
    
# Agents move
def get_agent_move(board):
    for i in range(0,9):
        if board[i] == " " and test_win_move(board, "X" ,i):
            return i
    for i in range(0,9):
        if board[i] == " " and test_win_move(board, "0", i):
            return i
        
   
    return win_strategy(board)
# Assemble the game
def tictactoe(board):
    ### Note you need to recreate your board again here if you wish to play the game more than once
    player = 1
    print("player1 is X \n player 2 is 0")
    while True:
        display_board(board)
        if (player % 2) != 0:
            print("player 1's choice")
            pos = int(input("enter position"))
            board[pos] = "X"
            if check_win("0", board):
                display_board(board)
                print("player " + str(player) + "won")
                break
            elif check_draw(board):
                display_board(board)
                print("draw")
                break
            player += 1
        else:
            print("player 2's choice")
            pos = int(get_agent_move(board))
            board[pos] = "0"

            if check_win("0", board):
                display_board(board)
                print("player " + str(player) + "won")
                
                break
            elif check_draw(board):
                display_board(board)
                print("draw")
                break
            player -= 1
    
    
    
# Play!!!
tictactoe(board)
class Tic_Tac_Toe:
    def __init__(self):
        pass
       

    def display_board(self, board):
        print(" %c | %c | %c " % (board[0], board[1], board[2]))
        print("___|___|___")
        print(" %c | %c | %c " % (board[3], board[4], board[5]))
        print("___|___|___")
        print(" %c | %c | %c " % (board[6], board[7], board[8]))
        print("   |   |   ")

    def check_win(self, player_mark, board):
        return (
            (board[0] == player_mark and board[1]== player_mark and board[2]==player_mark)or
            (board[3] == player_mark and board[4]== player_mark and board[5]==player_mark)or
            (board[6] == player_mark and board[7]== player_mark and board[8]==player_mark)or
            (board[1] == player_mark and board[4]== player_mark and board[7]==player_mark)or
            (board[0] == player_mark and board[3]== player_mark and board[6]==player_mark)or
            (board[2] == player_mark and board[5]== player_mark and board[8]==player_mark)or
            (board[0] == player_mark and board[4]== player_mark and board[8]==player_mark)or
            (board[2] == player_mark and board[4]== player_mark and board[6]==player_mark)
        )


    # Create a function to check its a Draw
    def check_draw(self,board):
        return " " not in board


    def board_copy(self,board):

        board1 = board.copy()

        return board1

    def test_win_move(self,board, player_mark, move):
        bcopy = self.board_copy(board)
        bcopy[move] = player_mark
        return self.check_win(player_mark, bcopy)


    def win_strategy(self,board):
        if board[4] == " ":
            return 4
        for i in [0, 2, 6, 8]:
            if board[i] == " ":
                return i
        for i in [1, 3, 5, 7]:
            if board[i] == " ":
                return i
    

    def get_agent_move(self, board):

        for i in range(0, 9):
            if board[i] == " " and self.test_win_move(board, "X", i):
                return i
        for i in range(0, 9):
            if board[i] == " " and self.test_win_move(board, "0", i):
                return i

        return self.win_strategy(board)


    def tictactoe(self, board):
    ### Note you need to recreate your board again here if you wish to play the game more than once
 
        player = 1
        print("player1 is X \n player 2 is 0")
        while True:
            self.display_board(board)
            if (player % 2) != 0:
                print("player 1's choice")
                pos = int(input("enter position: "))
                board[pos] = "X"
                if self.check_win("0", board):
                    self.display_board(board)
                    print("PLAYER " + str(player) + " WON!!!")
                    break
                elif self.check_draw(board):
                    self.display_board(board)
                    print("GAME DRAW!!!")
                    break
                player += 1
            else:
                print("player 2's choice")
                pos = int(self.get_agent_move(board))
                board[pos] = "0"

                if self.check_win("0", board):
                    self.display_board(board)
                    print("PLAYER " + str(player) + " WON!!!")
                    break
                elif self.check_draw(board):
                    self.display_board(board)
                    print("GAME DRAW!!!")
                    break
                player -= 1
            
            
board = [" "] * 9
obj = Tic_Tac_Toe()
obj.tictactoe(board)

