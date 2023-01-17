# State initialization.
import os

if not os.path.exists("./tape.txt"):
    open("./tape.txt", "w").write("['']")
    
if not os.path.exists("./tape_pos.txt"):
    open("./tape_pos.txt", "w").write("0")
    
if not os.path.exists("./stored_program_pos.txt"):
    open("./stored_program_pos.txt", "w").write("0")
# Fundamental Turing machine instructions.

def move_right():
    tape, tape_pos, stored_program_pos = get_state()
    tape_pos += 1
    if tape_pos >= len(tape):
        tape.append('')
        set_tape(tape)
    set_state(tape_pos, stored_program_pos + 1)        

def move_left():
    tape, tape_pos, stored_program_pos = get_state()
    tape_pos -= 1
    if tape_pos < 0:
        tape = [''] + tape
        set_tape(tape)
        tape_pos += 1
    set_state(tape_pos, stored_program_pos + 1)

def mark_and_move():
    tape, tape_pos, stored_program_pos = get_state()
    tape[tape_pos] = '*'
    set_tape(tape)
    move_right()
    set_state(tape_pos, stored_program_pos + 1)

def conditional_jmp(n):
    tape, tape_pos, stored_program_pos = get_state()
    if tape[tape_pos] == '*':
        set_state(updated_tape_pos, n)
    else:
        set_state(tape_pos, stored_program_pos + 1)

# State management and sample program.
import ast
    
def stored_program():
    jmp = conditional_jmp
    return [move_right, mark_and_move, move_left, lambda: jmp(0)]

def get_state():
    tape = ast.literal_eval(open("./tape.txt", "r").read())
    tape_pos = int(open("./tape_pos.txt" , "r").read())
    stored_program_pos = int(open("./stored_program_pos.txt", "r").read()) % 4
    return tape, tape_pos, stored_program_pos

def set_tape(tape):
    open("./tape.txt" , "w").write(str(tape))    

def set_state(tape_pos, stored_program_pos):
    open("./tape_pos.txt" , "w").write(str(tape_pos))
    open("./stored_program_pos.txt", "w").write(str(stored_program_pos))
    
def step_program():
    tape, tape_pos, stored_program_pos = get_state()
    stored_program_pos = int(open("./stored_program_pos.txt", "r").read()) % 4
    stored_program()[stored_program_pos]()
    print("""tape_pos: {0}
current_instruction: {1}
next_instruction: {2}""".format(tape_pos, 
                                stored_program()[stored_program_pos],
                                stored_program()[(stored_program_pos + 1) % 4]))
# Run one step in the program!
step_program()